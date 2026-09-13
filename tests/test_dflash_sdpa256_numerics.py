# SPDX-License-Identifier: Apache-2.0
"""Small native grouped-GQA parity checks; no checkpoint or unsafe allocation."""
import sys

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches import sdpa256_attention as sdpa


@pytest.mark.parametrize(
    "dtype_name,tolerance",
    [
        ("float32", 2e-4),
        ("float16", 3e-3),
        ("bfloat16", 2e-2),
    ],
)
@pytest.mark.parametrize("q_len", [4, 16])
@pytest.mark.parametrize("mask_kind", ["causal", "boolean", "additive"])
def test_grouped_gqa_bounded_numerical_parity(
    monkeypatch, dtype_name, tolerance, q_len, mask_kind
):
    from dflash_mlx.engine import gqa_sdpa
    from mlx_lm.models import base as lm_base
    from mlx_vlm.models import base as vlm_base

    # Preserve existing application/test-installed aliases and global state.
    for name, module in list(sys.modules.items()):
        if (
            module is not None
            and name.startswith(("mlx_lm.models.", "mlx_vlm.models.", "dflash_mlx."))
            and "scaled_dot_product_attention" in vars(module)
        ):
            monkeypatch.setattr(
                module,
                "scaled_dot_product_attention",
                module.scaled_dot_product_attention,
            )
    monkeypatch.setattr(sdpa, "_PATCHED", False)
    monkeypatch.setattr(sdpa, "_LM_SDPA256_WRAPPER", None)
    monkeypatch.setattr(sdpa, "_FORCE_TILED", None)
    monkeypatch.setattr(sdpa, "_SDPA256_MIN_KV_LEN", 32)
    monkeypatch.setenv("OMLX_SDPA256_TILED", "1")
    monkeypatch.setattr(sdpa, "_register_bounded_route", lambda _: True)
    # Rebind this test's helper to the exact current base before installation.
    monkeypatch.setattr(
        gqa_sdpa, "scaled_dot_product_attention", lm_base.scaled_dot_product_attention
    )
    native_bounded = sdpa._flash_sdpa256
    routed = []

    def observe(q, k, v, scale, mask, sinks=None):
        routed.append(q.shape)
        return native_bounded(q, k, v, scale, mask, sinks)

    monkeypatch.setattr(sdpa, "_flash_sdpa256", observe)
    assert sdpa.apply_sdpa256_attention_patch(min_kv_len=32)
    rng = np.random.default_rng(20260913)
    dtype = getattr(mx, dtype_name)
    q = mx.array(rng.normal(size=(1, 4, q_len, 256)).astype(np.float32), dtype=dtype)
    k = mx.array(rng.normal(size=(1, 1, 32, 256)).astype(np.float32), dtype=dtype)
    v = mx.array(rng.normal(size=(1, 1, 32, 256)).astype(np.float32), dtype=dtype)
    allowed = np.arange(32)[None, :] <= np.arange(32 - q_len, 32)[:, None]
    if mask_kind == "causal":
        mask = "causal"
    elif mask_kind == "boolean":
        mask = mx.array(allowed)
    else:
        mask = mx.array(np.where(allowed, 0.0, -1e9).astype(np.float32))
    with sdpa.scoped_unfused_headroom_provider(None):
        actual = gqa_sdpa.grouped_gqa_sdpa(q, k, v, scale=0.0625, mask=mask)
        mx.eval(actual)
    # Independent dense attention on the original, ungrouped head layout.
    q_np, k_np, v_np = [
        np.array(x.astype(mx.float32)).astype(np.float64) for x in (q, k, v)
    ]
    logits = q_np @ np.swapaxes(np.repeat(k_np, 4, axis=1), -1, -2) * 0.0625
    logits = np.where(allowed[None, None, :, :], logits, -np.inf)
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    expected = weights @ np.repeat(v_np, 4, axis=1)
    result = np.array(actual.astype(mx.float32))
    assert routed == [(1, 1, q_len * 4, 256)]
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, expected, atol=tolerance, rtol=tolerance)
