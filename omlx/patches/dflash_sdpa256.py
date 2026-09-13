# SPDX-License-Identifier: Apache-2.0
"""Request-scoped SDPA256 headroom for DFlash's scheduler-free executor.

This wires the existing bounded-attention route to DFlash's existing guard.
It does not change admission limits, attention kernels, or sampling.
"""

from collections.abc import Generator, Iterator
from typing import Any, TypeVar

import mlx.core as mx

from .sdpa256_attention import scoped_unfused_headroom_provider

_Event = TypeVar("_Event")
_UNBOUNDED_HEADROOM = (1 << 63) - 1


class _DFlashHeadroomProvider:
    """Read the active guard on the thread that actually runs model forwards."""

    def __init__(self, guard: Any) -> None:
        self._guard = guard

    def headroom(self) -> int:
        """Return bytes left for one unfused attention call, or -1 if unknown.

        Match Scheduler's target: the dynamic hard ceiling times its prefill
        safety fraction, capped by the stable abort limit times its margin.
        An explicitly disabled guard is different from limits not yet being
        propagated. Never interpret the latter as unlimited headroom.
        """
        guard = self._guard
        hard_cap = getattr(guard, "_memory_hard_limit_bytes", 0)
        if hard_cap <= 0:
            if (
                getattr(guard, "_memory_limits_propagated", False)
                and not getattr(guard, "_prefill_memory_guard", False)
            ):
                return _UNBOUNDED_HEADROOM
            return -1

        safety = getattr(guard, "_prefill_headroom_safety", 0.90)
        abort_limit = getattr(guard, "_memory_abort_limit_bytes", 0) or hard_cap
        abort_margin = getattr(guard, "_prefill_abort_margin", 0.90)
        target = min(int(hard_cap * safety), int(abort_limit * abort_margin))

        # Unlike preflight, this callback runs on the MLX executor. A sample
        # from request entry alone becomes stale as prefill grows the KV/cache.
        # Let probe failures propagate to SDPA256's bounded-on-error fallback;
        # do not substitute stale low usage and permit the unfused allocation.
        guard.record_mlx_active_memory(mx.get_active_memory())
        return target - guard._current_usage_bytes()


def dflash_sdpa256_events(
    events: Iterator[_Event], guard: Any | None
) -> Generator[_Event, None, None]:
    """Bind headroom during iteration, including generator cleanup.

    Both DFlash generation entrypoints consume this iterator synchronously on
    their executor. Merely creating an iterator on the event-loop thread must
    not bind a provider there. ``yield from`` forwards exceptions/close to the
    delegate before restoring the previous worker-local provider.

    A missing guard explicitly masks any previous provider, so a preceding
    engine cannot accidentally lend DFlash its more permissive headroom.
    """
    provider = _DFlashHeadroomProvider(guard) if guard is not None else None
    with scoped_unfused_headroom_provider(
        provider.headroom if provider is not None else None
    ):
        yield from events
