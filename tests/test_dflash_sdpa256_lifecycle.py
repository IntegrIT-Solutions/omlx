# SPDX-License-Identifier: Apache-2.0
"""Real async consumers and factory, with fake events instead of model forwards."""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from omlx import engine_core
from omlx.engine import dflash
from omlx.patches import sdpa256_attention as sdpa


class _Previous:
    def headroom(self):
        return 10**12


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("ending", ["normal", "error", "cancel"])
async def test_async_consumer_restores_policy_before_next_request(
    monkeypatch, streaming, ending
):
    from dflash_mlx import runtime
    from dflash_mlx.engine.events import TokenEvent
    from dflash_mlx.server.prefix_cache_flow import PrefixCacheFlow

    monkeypatch.setattr(sdpa, "_HEADROOM_PROVIDER_LOCAL", threading.local())
    engine = dflash.DFlashEngine(model_name="test", draft_model_path="test-draft")
    engine._loaded = True
    engine._tokenizer_obj = engine._executor_tokenizer = SimpleNamespace(
        decode=lambda *a, **k: "token", eos_token_ids=[2], eos_token_id=2,
    )
    engine._prefill_guard = dflash._DFlashPrefillGuard(Mock(), 2048)
    monkeypatch.setattr(engine, "_record_prefill_guard_active_memory", lambda: None)
    monkeypatch.setattr(engine, "_begin_runtime_cache_request", lambda: None)
    monkeypatch.setattr(engine, "_end_runtime_cache_request", lambda _: None)
    monkeypatch.setattr(engine, "_detect_needs_think_prefix", lambda _: False)
    monkeypatch.setattr(dflash, "create_streaming_detokenizer", lambda *a, **k: None)
    flow = SimpleNamespace(
        snapshot=None, snapshot_service=None, stable_prefix_len=None,
        cache_active=False, publish_generation_snapshot=False,
        hit_kind="miss", hit_tokens=0,
    )
    monkeypatch.setattr(PrefixCacheFlow, "for_request", classmethod(lambda cls, **kw: flow))
    blocked = threading.Event()
    release = threading.Event()
    observed = []
    mode = [ending]

    def fake_generate(**kwargs):
        selected = engine._prefill_guard
        assert sdpa._get_unfused_headroom_provider().__self__._guard is selected
        try:
            yield TokenEvent(token_id=7, generated_tokens=1,
                             acceptance_ratio=0.0, cycles_completed=1)
            if mode[0] == "cancel":
                blocked.set()
                assert release.wait(timeout=5), "cancel test did not release worker"
                yield object()  # Consumer sees its stop_event at this boundary.
            elif mode[0] == "error":
                raise RuntimeError("test generation error")
        finally:
            assert sdpa._get_unfused_headroom_provider().__self__._guard is selected
            observed.append((selected, threading.get_ident()))

    monkeypatch.setattr(runtime, "stream_dflash_generate", fake_generate)

    async def consume():
        if streaming:
            return [part async for part in engine.stream_generate([1], max_tokens=2)]
        return await engine.generate([1], max_tokens=2)

    previous = _Previous()
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(engine_core, "get_mlx_executor", lambda: executor)
        await loop.run_in_executor(executor, sdpa.set_unfused_headroom_provider,
                                   previous.headroom)
        try:
            task = asyncio.create_task(consume())
            if ending == "cancel":
                assert await asyncio.to_thread(blocked.wait, 5)
                task.cancel()
                await asyncio.sleep(0)  # Deliver cancellation before releasing work.
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=10)
            elif ending == "error" and not streaming:
                with pytest.raises(RuntimeError, match="test generation error"):
                    await asyncio.wait_for(task, timeout=10)
            else:
                result = await asyncio.wait_for(task, timeout=10)
                if ending == "error":
                    assert result[-1].finish_reason == "error"
            provider = await loop.run_in_executor(executor, sdpa._get_unfused_headroom_provider)
            assert provider.__self__ is previous
            assert observed and observed[0][1] != threading.get_ident()
            assert not engine.has_active_requests()

            # Reused worker, different guard, actual request submission again.
            mode[0] = "normal"
            engine._prefill_guard = dflash._DFlashPrefillGuard(Mock(), 2048)
            await asyncio.wait_for(consume(), timeout=10)
            assert observed[-1][0] is engine._prefill_guard
            assert observed[0][0] is not observed[-1][0]
            provider = await loop.run_in_executor(executor, sdpa._get_unfused_headroom_provider)
            assert provider.__self__ is previous
            assert not engine.has_active_requests()
        finally:
            release.set()
