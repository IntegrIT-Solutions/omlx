# SPDX-License-Identifier: Apache-2.0
"""Real installer interactions; shapes/dispatch are mocked, not Metal execution."""
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from omlx.patches import sdpa256_attention as sdpa
from omlx.patches import turboquant_attention as tq


@pytest.fixture
def runtime(monkeypatch):
    from mlx_lm.models import base as lm_base
    from mlx_vlm.models import base as vlm_base

    # Installers assign module attributes directly; undo all pre-existing
    # aliases as well as their flags so test order cannot contaminate a suite.
    aliases = [
        (module, module.scaled_dot_product_attention)
        for name, module in list(sys.modules.items())
        if module is not None
        and name.startswith(("mlx_lm.models.", "mlx_vlm.models.", "dflash_mlx."))
        and "scaled_dot_product_attention" in vars(module)
    ]
    for module, implementation in aliases:
        monkeypatch.setattr(module, "scaled_dot_product_attention", implementation)
    monkeypatch.setattr(sdpa, "_PATCHED", False)
    monkeypatch.setattr(sdpa, "_LM_SDPA256_WRAPPER", None)
    monkeypatch.setattr(sdpa, "_HEADROOM_PROVIDER_LOCAL", threading.local())
    monkeypatch.setattr(sdpa, "_FORCE_TILED", None)
    monkeypatch.setattr(sdpa, "_TILED_ROUTE_LOGGED", set())
    monkeypatch.setattr(sdpa, "_SDPA256_MIN_KV_LEN", 8192)
    monkeypatch.setattr(sdpa, "_SDPA256_MIN_Q_LEN", 16)
    monkeypatch.setenv("OMLX_SDPA256_TILED", "1")
    monkeypatch.setattr(sdpa, "_register_bounded_route", Mock())
    monkeypatch.setattr(tq, "_PATCHED", False)
    monkeypatch.setattr(tq, "_patch_update_eval_policy", Mock())
    monkeypatch.setattr(tq, "_patch_vlm_target_verify_attention", Mock())
    original = Mock(return_value="original")
    bounded = Mock(return_value="bounded")
    monkeypatch.setattr(lm_base, "scaled_dot_product_attention", original)
    monkeypatch.setattr(vlm_base, "scaled_dot_product_attention", original)
    monkeypatch.setattr(sdpa, "_flash_sdpa256", bounded)
    modules = []
    for name in ("dflash_mlx.engine.gqa_sdpa", "dflash_mlx.engine.target_qwen_gdn"):
        module = ModuleType(name)
        module.scaled_dot_product_attention = original
        monkeypatch.setitem(sys.modules, name, module)
        modules.append(module)
    return SimpleNamespace(base=lm_base, original=original, modules=modules, bounded=bounded)


def assert_bounded(runtime):
    q = SimpleNamespace(shape=(1, 1, 96, 256))
    k = v = SimpleNamespace(shape=(1, 1, 16384, 256))
    for module in [runtime.base, *runtime.modules]:
        assert module.scaled_dot_product_attention(
            q, k, v, cache=None, scale=0.0625, mask="causal"
        ) == "bounded"
    runtime.original.assert_not_called()


@pytest.mark.parametrize("capture", ["before_tq", "after_tq", "after_sdpa"])
def test_turboquant_before_sdpa_all_dflash_import_orders(runtime, capture):
    assert tq.apply_turboquant_attention_patch()
    if capture == "after_tq":
        for module in runtime.modules:
            module.scaled_dot_product_attention = runtime.base.scaled_dot_product_attention
    assert sdpa.apply_sdpa256_attention_patch()
    if capture == "after_sdpa":
        for module in runtime.modules:
            module.scaled_dot_product_attention = runtime.base.scaled_dot_product_attention
    assert_bounded(runtime)


def test_sdpa_before_turboquant_keeps_dflash_bounded(runtime):
    assert sdpa.apply_sdpa256_attention_patch()
    assert tq.apply_turboquant_attention_patch()
    assert_bounded(runtime)


def test_repeated_install_repairs_late_stale_alias_without_rewrapping(runtime):
    assert tq.apply_turboquant_attention_patch()
    assert sdpa.apply_sdpa256_attention_patch()
    installed = runtime.base.scaled_dot_product_attention
    for module in runtime.modules:
        module.scaled_dot_product_attention = runtime.original
    assert sdpa.apply_sdpa256_attention_patch() is False
    assert runtime.base.scaled_dot_product_attention is installed
    assert_bounded(runtime)


def test_early_dflash_opt_out_then_tq_then_enable(runtime):
    # Opt-out deliberately does not call the installer; imported aliases remain.
    assert sdpa._PATCHED is False
    assert tq.apply_turboquant_attention_patch()
    assert sdpa.apply_sdpa256_attention_patch()
    assert sdpa.apply_sdpa256_attention_patch() is False
    assert_bounded(runtime)


def test_repair_does_not_remove_later_outer_base_wrapper(runtime):
    assert sdpa.apply_sdpa256_attention_patch()
    assert tq.apply_turboquant_attention_patch()
    outer = runtime.base.scaled_dot_product_attention
    runtime.modules[0].scaled_dot_product_attention = runtime.original
    assert sdpa.apply_sdpa256_attention_patch() is False
    assert runtime.base.scaled_dot_product_attention is outer
    assert_bounded(runtime)


def test_genuinely_custom_alias_is_preserved_even_after_repeated_install(runtime):
    custom = Mock(return_value="custom")
    runtime.modules[0].scaled_dot_product_attention = custom
    assert tq.apply_turboquant_attention_patch()
    assert sdpa.apply_sdpa256_attention_patch()
    assert sdpa.apply_sdpa256_attention_patch() is False
    assert runtime.modules[0].scaled_dot_product_attention is custom


def test_unrelated_module_is_not_rebound(runtime, monkeypatch):
    unrelated = ModuleType("unrelated_attention_test")
    unrelated.scaled_dot_product_attention = runtime.original
    monkeypatch.setitem(sys.modules, unrelated.__name__, unrelated)
    assert tq.apply_turboquant_attention_patch()
    assert sdpa.apply_sdpa256_attention_patch()
    assert unrelated.scaled_dot_product_attention is runtime.original


def test_concrete_provenance_does_not_query_dynamic_attributes(runtime):
    # Mock manufactures attributes through __getattr__; absent provenance must
    # terminate traversal rather than inventing an infinite chain of mocks.
    assert tq.apply_turboquant_attention_patch()
    assert sdpa.apply_sdpa256_attention_patch()
    assert "_omlx_sdpa_original" not in vars(runtime.original)
    assert_bounded(runtime)


def test_provenance_cycle_terminates(runtime):
    assert tq.apply_turboquant_attention_patch()
    outer = runtime.base.scaled_dot_product_attention
    runtime.original._omlx_sdpa_original = outer
    assert sdpa.apply_sdpa256_attention_patch()
    assert sdpa.apply_sdpa256_attention_patch() is False
    assert_bounded(runtime)


def test_forced_unfused_opt_out_is_preserved(runtime, monkeypatch):
    monkeypatch.setenv("OMLX_SDPA256_TILED", "0")
    assert tq.apply_turboquant_attention_patch()
    assert sdpa.apply_sdpa256_attention_patch()
    q = SimpleNamespace(shape=(1, 1, 96, 256))
    k = v = SimpleNamespace(shape=(1, 1, 16384, 256))
    assert runtime.modules[0].scaled_dot_product_attention(
        q, k, v, cache=None, scale=0.0625, mask="causal"
    ) == "original"
    runtime.bounded.assert_not_called()
