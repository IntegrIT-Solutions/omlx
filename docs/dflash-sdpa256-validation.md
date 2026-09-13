# Validating request-scoped SDPA256 routing in DFlash

This change addresses DFlash's bounded-attention installation and provider
ownership. It is not a guarantee that an arbitrary context fits on a 48 GiB Mac.
The draft and target weights, KV caches, prefix snapshots, explicit attention
masks, allocator pools, and other applications still need memory. Existing
admission checks and runtime memory limits remain in effect.

## Regression tests

In an isolated development checkout with the project's pinned dependencies:

```bash
python -m pytest \
  tests/test_dflash_sdpa256.py \
  tests/test_sdpa256_attention.py \
  tests/test_dflash_prefill_memory_guard.py \
  tests/test_dflash_engine.py \
  tests/test_process_memory_enforcer.py -q

python -m pytest -m "not slow and not integration"
```

The new tests deliberately use small or shape-only inputs. They do not attempt
to reproduce an out-of-memory condition by exhausting a machine. The grouped
GQA test checks dispatch through the pinned DFlash dependency, not numerical
accuracy of a new kernel: this patch introduces no kernel.

Before publishing test counts, distinguish the complete upstream pytest run
from any isolated harness that replaces MLX with stubs. An isolated routing test
is not evidence that a Metal kernel executed, that the application built, or
that a 48 GiB workload is safe.

## Clean-process and lifecycle matrix

Validate these paths with both normal and streaming completions:

1. Start a fresh process, load DFlash first, and make a short request. Confirm
   the SDPA256 installer ran without requiring a Batched/VLM engine first.
2. Use an eligible head-dim-256 prompt with limited but safe headroom. Confirm
   the existing bounded route is selected. Keep the guard enabled.
3. Complete a request, cancel another, then make a follow-up. The active
   provider must belong to the current DFlash guard and must not remain bound
   after its event iterator is closed.
4. Load another engine first, then DFlash; also test DFlash stop/reload and
   transition to its configured fallback engine. No provider may be borrowed
   from a preceding request, even on a reused executor thread.
5. Check missing/unpropagated guard state using mocks: it must select the
   bounded default, not unlimited headroom. An explicit guard opt-out remains
   an opt-out; do not use that mode for memory-safety measurements.

`sdpa256_prefill_enabled=False` prevents this engine from installing the
process-wide patch. As with the existing Batched/VLM convention, it does not
uninstall a wrapper another engine already installed. This change does not add
per-model Auto/Tiled/Unfused policy or change `OMLX_SDPA256_TILED` semantics.

## Apple Silicon hardware check

Keep the known-working installation and its settings intact. Use a separate
checkout, virtual environment, settings/cache directory, and unused loopback
port. Do not run the stock and experimental models concurrently during the
comparison. Do not point the experimental build at the production cache.

Record the exact oMLX commit, MLX and dflash-mlx versions, macOS version, hardware,
target and draft revisions/quantization, effective guard limits, prefix-cache
settings, and current kernel wired-memory setting. Do not change the kernel
limit, guard tier, ANE, quantization, and decoder all at once.

Suggested target/draft pair for the reported use case:

- `Qwen3.8-27B-oQ4e-mtp` as target, using DFlash rather than native MTP.
- `z-lab/Qwen3.8-27B-DFlash2` as draft, with the actual draft quantization recorded.

Use one request at a time. Begin with short prompts and short bounded output.
Only advance from 8K to 16K, 32K, and 48K when the preceding run has adequate
headroom, remains responsive, and finishes cleanly. These are test stages, not
promised supported context lengths. Stop on sustained memory pressure, rapid
swap growth, repeated memory aborts, or unexpected allocation growth; do not
intentionally drive the Mac into OOM to demonstrate a fix.

For each stage, measure a cold request and an append-only follow-up. Avoid
requesting an intentionally dangerous unpatched long-context baseline; routing
regression tests can establish the missing-hook behavior without that risk.

| Context | Cold TTFT / prefill rate | Follow-up cached tokens / TTFT | Decode rate | Peak MLX active / process footprint | Guard outcome / SDPA route |
|---|---|---|---|---|---|
| 8K | Not measured | Not measured | Not measured | Not measured | Not measured |
| 16K | Not measured | Not measured | Not measured | Not measured | Not measured |
| 32K | Not measured | Not measured | Not measured | Not measured | Not measured |
| 48K | Not measured | Not measured | Not measured | Not measured | Not measured |

Compare DFlash and Lightning MTP on the **same patched oMLX commit**, same
workload and context, matching sampling settings, and the same machine state.
Report total task time as well as token throughput. Faster decoding alone can
be offset by slower prefill, snapshot handling, or a larger resident footprint.
Do not assume bit-identical generated text across attention implementations.

Existing route logs are rate-limited by reason, so their absence on a later
request is not proof that a route stopped running. Tests instrument the dispatch
point directly; hardware runs should combine logs with memory and timing data.

## Scope of a successful result

A successful result establishes that eligible DFlash attention reaches the
existing bounded implementation with the correct current guard. It does not
resolve every memory-accounting or Auto-policy concern in issue #3241, nor does
it establish a DFlash speedup over Lightning MTP. Keep the PR draft until the
native integration and lifecycle checks above have been reviewed.


## Review follow-up: retained pre-wrapper aliases

SDPA256 now distinguishes base-wrapper installation from DFlash alias coverage.
TurboQuant and FA256 record their original callables using private oMLX provenance metadata.
SDPA256 reconciles DFlash aliases against that identity chain on both first and
repeated installation. A legitimate reference captured before TurboQuant is
repaired; a genuinely custom function is not replaced. The base wrapper is not
stacked again, and a later outer wrapper on the base is left intact.

Run the additional regression tests alongside the suites above:

```bash
python -m pytest -q tests/test_dflash_sdpa256_load_order.py \
  tests/test_dflash_sdpa256_lifecycle.py tests/test_dflash_sdpa256_numerics.py
```

The load-order suite uses the actual TurboQuant and SDPA256 installers with
shape-only inputs and an instrumented bounded dispatch. It covers pre-import,
late import, repeated installation, an explicit early opt-out, provenance
cycles, both installation orders, and preservation of unrelated custom code.
It must not be described as a Metal or numerical test.

The lifecycle tests exercise both real engine consumer entry points with fake
DFlash events and asynchronous cancellation, checking the provider on the worker
and on a subsequent request. The numerical tests use the actual pinned grouped
GQA helper and bounded attention implementation with small arrays. They compare
against an independent dense reference for causal, Boolean, and additive masks,
including verification shapes that become eligible only after GQA reshaping.

Native dependency tests, kernel execution, and 48 GiB hardware results must be
recorded separately. An isolated host-side extraction of repository functions
is not a substitute for running these complete files with pinned dependencies.
