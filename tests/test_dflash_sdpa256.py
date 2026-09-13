# SPDX-License-Identifier: Apache-2.0
"""DFlash's SDPA256 route must not depend on an earlier scheduler engine.

No model weights or unsafe large allocations are required. Hardware benchmarks
remain a separate integration gate; these tests cover routing and ownership.
"""

import asyncio
import gc
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from omlx.patches import sdpa256_attention as sdpa
from omlx.patches.dflash_sdpa256 import (
    _DFlashHeadroomProvider,
    dflash_sdpa256_events,
)


class _Guard:
    def __init__(self, hard=1000, physical=100):
        self._memory_hard_limit_bytes = hard
        self._memory_abort_limit_bytes = 0
        self._prefill_headroom_safety = 0.90
        self._prefill_abort_margin = 0.90
        self._memory_limits_propagated = True
        self._prefill_memory_guard = True
        self._last_mlx_active_memory_bytes = 0
        self.physical = physical

    def record_mlx_active_memory(self, value):
        self._last_mlx_active_memory_bytes = value

    def _current_usage_bytes(self):
        return max(self.physical, self._last_mlx_active_memory_bytes)


class _Owner:
    def __init__(self, value):
        self.value = value

    def headroom(self):
        return self.value


@pytest.fixture(autouse=True)
def isolated_provider(monkeypatch):
    monkeypatch.setattr(sdpa, "_LM_SDPA256_WRAPPER", sdpa._LM_SDPA256_WRAPPER)
    monkeypatch.delenv("OMLX_SDPA256_TILED", raising=False)
    monkeypatch.setattr(sdpa, "_HEADROOM_PROVIDER_LOCAL", threading.local())
    monkeypatch.setattr(sdpa, "_FORCE_TILED", None)
    monkeypatch.setattr(sdpa, "_TILED_ROUTE_LOGGED", set())
    monkeypatch.setattr(sdpa, "_SDPA256_MIN_KV_LEN", 8192)
    monkeypatch.setattr(sdpa, "_SDPA256_MIN_Q_LEN", 16)


@pytest.fixture
def active_memory(monkeypatch):
    sample = Mock(return_value=100)
    monkeypatch.setattr("omlx.patches.dflash_sdpa256.mx.get_active_memory", sample)
    return sample


def test_unknown_limits_are_not_an_explicit_opt_out(active_memory):
    guard = _Guard(hard=0)
    guard._memory_limits_propagated = False
    guard._prefill_memory_guard = False
    assert _DFlashHeadroomProvider(guard).headroom() == -1
    active_memory.assert_not_called()


def test_unpropagated_old_guard_is_bounded(active_memory):
    guard = SimpleNamespace(_memory_hard_limit_bytes=0)
    assert _DFlashHeadroomProvider(guard).headroom() == -1
    active_memory.assert_not_called()


def test_propagated_guard_without_ceiling_is_bounded(active_memory):
    guard = _Guard(hard=0)
    assert _DFlashHeadroomProvider(guard).headroom() == -1


def test_explicitly_disabled_guard_keeps_existing_opt_out(active_memory):
    guard = _Guard(hard=0)
    guard._prefill_memory_guard = False
    assert _DFlashHeadroomProvider(guard).headroom() > 10**18
    active_memory.assert_not_called()


@pytest.mark.parametrize(
    "hard,abort,safety,margin,used,expected",
    [
        (1000, 0, 0.90, 0.90, 100, 800),
        (1000, 800, 0.90, 0.90, 100, 620),
        (800, 1000, 0.90, 0.90, 100, 620),
        (1000, 0, 0.925, 0.95, 100, 825),
        (1000, 800, 0.925, 0.95, 100, 660),
        (1000, 0, 0.90, 0.90, 950, -50),
    ],
)
def test_headroom_uses_both_limits_and_tier_margins(
    active_memory, hard, abort, safety, margin, used, expected
):
    guard = _Guard(hard=hard, physical=used)
    guard._memory_abort_limit_bytes = abort
    guard._prefill_headroom_safety = safety
    guard._prefill_abort_margin = margin
    assert _DFlashHeadroomProvider(guard).headroom() == expected


def test_headroom_resamples_active_memory_for_each_attention_call(active_memory):
    guard = _Guard()
    active_memory.side_effect = [100, 850]
    provider = _DFlashHeadroomProvider(guard)
    assert provider.headroom() == 800
    assert provider.headroom() == 50
    assert guard._last_mlx_active_memory_bytes == 850


def test_headroom_reads_new_limits_without_recreating_provider(active_memory):
    guard = _Guard()
    provider = _DFlashHeadroomProvider(guard)
    assert provider.headroom() == 800
    guard._memory_hard_limit_bytes = 500
    assert provider.headroom() == 350


def _attention_shapes(q_len=128, kv_len=16384, head_dim=256):
    return (
        SimpleNamespace(shape=(1, 24, q_len, head_dim)),
        SimpleNamespace(shape=(1, 4, kv_len, head_dim)),
    )


@pytest.mark.parametrize("headroom,expected", [(0, True), (10**12, False)])
def test_live_headroom_controls_existing_bounded_route(headroom, expected):
    owner = _Owner(headroom)
    q, k = _attention_shapes()
    with sdpa.scoped_unfused_headroom_provider(owner.headroom):
        assert sdpa._should_route(q, k, None, "causal", None) is expected


def test_failed_live_probe_uses_bounded_route_not_stale_sample(active_memory):
    active_memory.side_effect = RuntimeError("probe failed")
    guard = _Guard(hard=10**12)
    provider = _DFlashHeadroomProvider(guard)
    q, k = _attention_shapes()
    with sdpa.scoped_unfused_headroom_provider(provider.headroom):
        assert sdpa._should_route(q, k, None, "causal", None) is True


@pytest.mark.parametrize(
    "q_len,kv_len,head_dim,cache",
    [
        (1, 16384, 256, None),
        (8, 16384, 256, None),
        (128, 8191, 256, None),
        (128, 16384, 128, None),
        (128, 16384, 256, SimpleNamespace(bits=4)),
    ],
)
def test_decode_small_context_other_heads_and_quantized_kv_stay_out(
    q_len, kv_len, head_dim, cache
):
    q, k = _attention_shapes(q_len, kv_len, head_dim)
    with sdpa.scoped_unfused_headroom_provider(None):
        assert sdpa._should_route(q, k, cache, "causal", None) is False


@pytest.mark.parametrize("forced,expected", [(True, True), (False, False)])
def test_existing_force_override_is_respected(monkeypatch, forced, expected):
    monkeypatch.setattr(sdpa, "_FORCE_TILED", forced)
    q, k = _attention_shapes()
    with sdpa.scoped_unfused_headroom_provider(None):
        assert sdpa._should_route(q, k, None, "causal", None) is expected


def test_scope_nests_and_masks_previous_provider():
    previous, current = _Owner(1000), _Owner(100)
    sdpa.set_unfused_headroom_provider(previous.headroom)
    with sdpa.scoped_unfused_headroom_provider(current.headroom):
        assert sdpa._get_unfused_headroom_provider().__self__ is current
        with sdpa.scoped_unfused_headroom_provider(None):
            assert sdpa._get_unfused_headroom_provider() is None
        assert sdpa._get_unfused_headroom_provider().__self__ is current
    assert sdpa._get_unfused_headroom_provider().__self__ is previous


def test_scope_does_not_retain_previous_engine():
    previous = _Owner(1000)
    ref = weakref.ref(previous)
    sdpa.set_unfused_headroom_provider(previous.headroom)
    with sdpa.scoped_unfused_headroom_provider(None):
        del previous
        gc.collect()
        assert ref() is None
    assert sdpa._get_unfused_headroom_provider() is None


@pytest.mark.parametrize("ending", ["finish", "error", "close"])
def test_event_scope_restores_provider_after_delegate_cleanup(ending, active_memory):
    previous = _Owner(10**12)
    sdpa.set_unfused_headroom_provider(previous.headroom)
    guard = _Guard()
    observed = []

    def events():
        try:
            assert sdpa._get_unfused_headroom_provider().__self__._guard is guard
            yield "first"
            if ending == "error":
                raise RuntimeError("generation failed")
            yield "last"
        finally:
            # Cleanup is still on the active request's policy, not its predecessor.
            observed.append(sdpa._get_unfused_headroom_provider().__self__._guard)

    iterator = dflash_sdpa256_events(events(), guard)
    assert sdpa._get_unfused_headroom_provider().__self__ is previous
    assert next(iterator) == "first"
    if ending == "close":
        iterator.close()
    elif ending == "error":
        with pytest.raises(RuntimeError, match="generation failed"):
            next(iterator)
    else:
        assert list(iterator) == ["last"]
    assert observed == [guard]
    assert sdpa._get_unfused_headroom_provider().__self__ is previous


def test_missing_guard_cannot_borrow_previous_engine_headroom():
    previous = _Owner(10**12)
    sdpa.set_unfused_headroom_provider(previous.headroom)

    def events():
        assert sdpa._get_unfused_headroom_provider() is None
        q, k = _attention_shapes()
        assert sdpa._should_route(q, k, None, "causal", None)
        yield 1

    assert list(dflash_sdpa256_events(events(), None)) == [1]
    assert sdpa._get_unfused_headroom_provider().__self__ is previous


def test_iterator_binds_on_consumption_thread_not_creation_thread():
    main_owner, worker_owner = _Owner(123), _Owner(456)
    sdpa.set_unfused_headroom_provider(main_owner.headroom)

    def events():
        assert sdpa._get_unfused_headroom_provider() is None
        yield threading.get_ident()

    iterator = dflash_sdpa256_events(events(), None)

    def consume():
        sdpa.set_unfused_headroom_provider(worker_owner.headroom)
        ids = list(iterator)
        assert sdpa._get_unfused_headroom_provider().__self__ is worker_owner
        return ids

    with ThreadPoolExecutor(max_workers=1) as executor:
        ids = executor.submit(consume).result(timeout=5)
    assert ids[0] != threading.get_ident()
    assert sdpa._get_unfused_headroom_provider().__self__ is main_owner


def test_two_engine_policies_remain_thread_local():
    barrier = threading.Barrier(2)

    def worker(value):
        owner = _Owner(value)
        with sdpa.scoped_unfused_headroom_provider(owner.headroom):
            barrier.wait(timeout=5)
            assert sdpa._get_unfused_headroom_provider()() == value
        assert sdpa._get_unfused_headroom_provider() is None

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(worker, 1)
        second = executor.submit(worker, 2)
        first.result(timeout=10)
        second.result(timeout=10)


def test_installer_rebinds_captured_dflash_aliases_without_clobbering_custom(
    monkeypatch,
):
    from mlx_lm.models import base as lm_base
    from mlx_vlm.models import base as vlm_base

    original = Mock(name="original_lm_sdpa")
    custom = Mock(name="custom_sdpa")
    monkeypatch.setattr(lm_base, "scaled_dot_product_attention", original)
    monkeypatch.setattr(vlm_base, "scaled_dot_product_attention", Mock())
    monkeypatch.setattr(sdpa, "_PATCHED", False)
    monkeypatch.setattr(sdpa, "_register_bounded_route", Mock())
    modules = {}
    for name, fn in [
        ("dflash_mlx.engine.gqa_sdpa", original),
        ("dflash_mlx.engine.target_qwen_gdn", original),
        ("dflash_mlx.engine.custom_test", custom),
        ("unrelated_sdpa_test", original),
    ]:
        module = ModuleType(name)
        module.scaled_dot_product_attention = fn
        monkeypatch.setitem(sys.modules, name, module)
        modules[name] = module

    assert sdpa.apply_sdpa256_attention_patch() is True
    patched = lm_base.scaled_dot_product_attention
    assert patched is not original
    assert modules["dflash_mlx.engine.gqa_sdpa"].scaled_dot_product_attention is patched
    assert (
        modules["dflash_mlx.engine.target_qwen_gdn"].scaled_dot_product_attention
        is patched
    )
    assert (
        modules["dflash_mlx.engine.custom_test"].scaled_dot_product_attention is custom
    )
    assert modules["unrelated_sdpa_test"].scaled_dot_product_attention is original
    sdpa.apply_sdpa256_attention_patch()
    assert lm_base.scaled_dot_product_attention is patched


@pytest.mark.parametrize("enabled", [True, False])
def test_start_installs_route_before_loading_target(monkeypatch, enabled):
    from omlx import engine_core
    from omlx.engine import dflash

    class LoadReached(Exception):
        pass

    calls = []
    monkeypatch.setattr(
        sdpa, "apply_sdpa256_attention_patch", lambda: calls.append("sdpa")
    )
    monkeypatch.setattr(dflash, "maybe_apply_pre_load_patches", lambda *a, **k: None)

    def load_target(*args, **kwargs):
        assert calls == (["sdpa"] if enabled else [])
        calls.append("target")
        raise LoadReached

    stubs = {
        "dflash_mlx.draft_backend": dict(EagerDraftBackend=object),
        "dflash_mlx.engine.target_ops": dict(bind_draft_to_target=lambda *a, **k: None),
        "dflash_mlx.runtime.loading": dict(
            load_target_bundle=load_target, load_draft_bundle=Mock()
        ),
        "omlx.patches.dflash_laguna": dict(install_dflash_laguna_backend=lambda: None),
        "omlx.patches.dflash_lifecycle": dict(
            install_dflash_lifecycle_wrap=lambda: None
        ),
    }
    for name, attrs in stubs.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    engine = object.__new__(dflash.DFlashEngine)
    engine._loaded = False
    engine._model_name = "no-model-download"
    engine._model_settings = SimpleNamespace(sdpa256_prefill_enabled=enabled)
    engine._build_runtime_context = lambda: SimpleNamespace(runtime=SimpleNamespace())

    async def exercise():
        with pytest.raises(LoadReached):
            await engine.start()

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(engine_core, "get_mlx_executor", lambda: executor)
        asyncio.run(exercise())
    assert calls == (["sdpa", "target"] if enabled else ["target"])


def test_delegate_close_failure_still_restores_provider():
    previous = _Owner(10**12)
    sdpa.set_unfused_headroom_provider(previous.headroom)

    def events():
        try:
            yield 1
        finally:
            raise RuntimeError("cleanup failed")

    iterator = dflash_sdpa256_events(events(), None)
    assert next(iterator) == 1
    with pytest.raises(RuntimeError, match="cleanup failed"):
        iterator.close()
    assert sdpa._get_unfused_headroom_provider().__self__ is previous


def test_invalid_provider_does_not_overwrite_previous_provider():
    previous = _Owner(10**12)
    sdpa.set_unfused_headroom_provider(previous.headroom)
    with pytest.raises((AttributeError, TypeError)):
        with sdpa.scoped_unfused_headroom_provider(lambda: 0):
            pytest.fail("a non-bound provider must not be accepted")
    assert sdpa._get_unfused_headroom_provider().__self__ is previous


def test_grouped_dflash_prefill_reaches_bounded_route(monkeypatch):
    import mlx.core as mx
    from dflash_mlx.engine import gqa_sdpa
    from mlx_lm.models import base as lm_base
    from mlx_vlm.models import base as vlm_base

    original = Mock(side_effect=AssertionError("unsafe unpatched route"))
    monkeypatch.setattr(lm_base, "scaled_dot_product_attention", original)
    monkeypatch.setattr(vlm_base, "scaled_dot_product_attention", Mock())
    monkeypatch.setattr(gqa_sdpa, "scaled_dot_product_attention", original)
    monkeypatch.setattr(sdpa, "_PATCHED", False)
    monkeypatch.setattr(sdpa, "_register_bounded_route", Mock())
    monkeypatch.setattr(sdpa, "_SDPA256_MIN_KV_LEN", 32)
    calls = []

    def bounded(q, k, v, scale, mask, sinks=None):
        calls.append((q.shape, mask))
        return q

    monkeypatch.setattr(sdpa, "_flash_sdpa256", bounded)
    sdpa.apply_sdpa256_attention_patch(min_kv_len=32)
    q = mx.zeros((1, 4, 16, 256), dtype=mx.float16)
    k = v = mx.zeros((1, 2, 32, 256), dtype=mx.float16)
    with sdpa.scoped_unfused_headroom_provider(None):
        result = gqa_sdpa.grouped_gqa_sdpa(q, k, v, scale=0.0625, mask="causal")
    assert result.shape == q.shape
    assert calls[0][0] == (1, 2, 32, 256)
    assert isinstance(calls[0][1], mx.array)
    original.assert_not_called()


def test_engine_event_factory_wraps_the_real_consumer_boundary(monkeypatch):
    from omlx.engine.dflash import DFlashEngine

    guard = _Guard()
    previous = _Owner(10**12)
    sdpa.set_unfused_headroom_provider(previous.headroom)
    flow = SimpleNamespace(
        snapshot=None,
        snapshot_service=None,
        stable_prefix_len=None,
        cache_active=False,
        publish_generation_snapshot=False,
        hit_kind="miss",
    )
    seen = []

    def generate(**kwargs):
        seen.append(sdpa._get_unfused_headroom_provider().__self__._guard)
        yield "token"

    runtime = ModuleType("dflash_mlx.runtime")
    runtime.get_stop_token_ids = lambda _: {2}
    runtime.stream_dflash_generate = generate
    monkeypatch.setitem(sys.modules, "dflash_mlx.runtime", runtime)
    prefix = ModuleType("dflash_mlx.server.prefix_cache_flow")
    prefix.PrefixCacheFlow = SimpleNamespace(for_request=lambda **kwargs: flow)
    monkeypatch.setitem(sys.modules, prefix.__name__, prefix)

    engine = object.__new__(DFlashEngine)
    for name, value in {
        "_prefill_guard": guard,
        "_model_name": "test",
        "_draft_model_path": "draft",
        "_target_model": None,
        "_target_ops": None,
        "_executor_tokenizer": None,
        "_draft_model": None,
        "_draft_backend": None,
        "_runtime_context": None,
        "_suppress_token_ids": set(),
        "_block_size": 16,
    }.items():
        setattr(engine, name, value)
    iterator, actual_flow, stop_ids = engine._stream_dflash_events([1], 2)
    assert sdpa._get_unfused_headroom_provider().__self__ is previous
    assert list(iterator) == ["token"]
    assert seen == [guard]
    assert actual_flow is flow
    assert stop_ids == {2}
    assert sdpa._get_unfused_headroom_provider().__self__ is previous
