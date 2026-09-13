# Local DFlash A/B validation runbook

This document is intentionally safe for a public repository. Do not add private hostnames, IP addresses, API keys, account identifiers, client/project names, private source code, user names, serial numbers, or local filesystem paths that reveal personal information. If logs contain any of those, sanitize them before posting.

## Goal

Determine whether DFlash is a worthwhile daily-driver replacement for Lightning MTP for Qwen3.8-27B on a 48 GiB Apple Silicon Mac.

The question is not only whether DFlash works. The decision must weigh:

1. decode speed,
2. prefill and TTFT,
3. memory footprint and guard behavior,
4. stable long-context capacity,
5. prefix-cache / continuation behavior,
6. real coding-agent responsiveness.

A speedup is not sufficient if the practical raw context collapses materially.

Target hardware class for this experiment:

- Apple Silicon M4 Pro
- 48 GiB unified memory
- one local inference request at a time

Do not publish the machine hostname, serial number, local user name, IP address, Tailnet identity, API keys, or any unrelated local configuration.

## Branch and commits under test

Use the public fork branch:

```text
fix/dflash-sdpa256-headroom
```

The two important commits are:

```text
7fe063e9f3c0c88c155c9b901e9113022d10063c
  fix(dflash): scope SDPA256 prefill routing to active guard

dba7c10c678df6354476964c66c1a84a41608779
  fix(dflash): repair retained SDPA aliases across wrapper load orders
```

Benchmark the branch tip unless it has moved unexpectedly. Record the exact SHA used.

Before running performance tests, independently review both commits and the surrounding code. Do not assume the existing review is correct merely because tests pass.

---

# 1. What the PR fixes

## 1.1 Original DFlash memory-safety gap

DFlash bypasses the normal scheduler-based engine path. BatchedEngine and VLMBatchedEngine already installed oMLX's head-dim-256 bounded SDPA prefill route and supplied live memory-headroom information to that route. DFlash did not.

For Qwen3.8-27B this matters because an eligible long-context head-dim-256 prefill can otherwise fall through to an unfused attention path whose transient score matrix grows approximately O(L^2). On a memory-constrained Mac this can cause avoidable prefill pressure even though oMLX already contains a bounded/tiled implementation.

The first commit adds the DFlash integration:

- DFlash installs the existing SDPA256 bounded-prefill patch during engine startup.
- DFlash exposes live prefill headroom from its existing `_DFlashPrefillGuard`.
- The headroom provider is bound on the actual inference/MLX worker while a DFlash request is being consumed.
- The previous provider is restored after completion, exception, cancellation, or generator close.
- Missing or not-yet-propagated guard state defaults to the bounded/safe route rather than borrowing stale permissive state from a prior request.
- An explicitly disabled guard remains an explicit opt-out.

This is an integration fix. It does not remove target weights, draft weights, KV state, prefix snapshots, masks, allocator pools, or other memory consumers.

## 1.2 Retained-alias/load-order bug found during review

A second independent review found a real load-order bug in the first implementation.

DFlash's pinned dependency imports `scaled_dot_product_attention` into helper modules such as its grouped-GQA and target-Qwen paths. Those modules can retain an older function reference even after another oMLX patch replaces the base mlx-lm attention function.

A concrete problematic sequence was:

```text
DFlash helper captures original attention function A
        ↓
TurboQuant or FA256 wraps base attention: B -> A
        ↓
SDPA256 wraps current base: C -> B -> A
        ↓
DFlash helper still points directly to A
```

The first fix only rebound DFlash aliases when they were identical to the function SDPA256 had just wrapped. Therefore a legitimate older reference such as `A` could be skipped while the base path was correctly wrapped.

The second commit fixes that by recording explicit oMLX wrapper provenance:

- TurboQuant wrappers record their original callable.
- FA256 wrappers record their original callable.
- SDPA256 records its original callable.
- SDPA256 can walk the explicit provenance chain to determine which older references are legitimate oMLX ancestors.
- Known DFlash aliases pointing at one of those legitimate ancestors are repaired.
- Reconciliation also runs on repeated SDPA256 installation, so a late-imported or previously missed DFlash alias can be repaired without stacking another base wrapper.
- Unrelated custom implementations are not overwritten.
- A later outer wrapper on the base is not removed.
- Provenance traversal is cycle-safe and uses concrete function metadata rather than arbitrary closure inspection.

## 1.3 Existing validation already completed

The follow-up validation run used macOS ARM64 with real pinned dependencies and Metal available. The focused suite completed with:

```text
428 passed
```

Coverage included:

- retained-alias and installer-order cases,
- TurboQuant / SDPA256 orderings,
- FA256 wrapper chains,
- repeated installation,
- preservation of custom functions,
- real async DFlash consumer completion/error/cancellation paths with fake model events,
- grouped-GQA numerical parity against an independent dense reference for causal, Boolean, and additive masks,
- existing DFlash engine, SDPA256, memory-guard, and process-memory-enforcer tests.

This does **not** prove full-model long-context stability on a 48 GiB machine and does **not** prove a speedup.

---

# 2. Staff-level review checklist before benchmarking

A local agent must review the branch before building.

At minimum inspect:

```text
omlx/engine/dflash.py
omlx/patches/dflash_sdpa256.py
omlx/patches/sdpa256_attention.py
omlx/patches/turboquant_attention.py
omlx/patches/qwen35_fa256_attention.py
omlx/process_memory_enforcer.py
omlx/model_settings.py

tests/test_dflash_sdpa256.py
tests/test_dflash_sdpa256_load_order.py
tests/test_dflash_sdpa256_lifecycle.py
tests/test_dflash_sdpa256_numerics.py

docs/dflash-sdpa256-validation.md
```

Review specifically for:

- stale alias ownership across engine load/unload/reload,
- wrapper ordering with TurboQuant and FA256,
- accidental replacement of custom attention implementations,
- provider leakage between requests on reused executor threads,
- provider cleanup under cancellation and exceptions,
- guard semantics before memory limits have propagated,
- guard semantics when explicitly disabled,
- whether DFlash event iteration happens entirely inside the provider scope,
- whether generator close/cleanup can occur after the provider has already been restored,
- whether the bounded route's memory estimate matches the guard accounting used for admission,
- any process-global state that should instead be engine- or worker-scoped.

If a new correctness bug is found, stop the benchmark and fix/review it first.

---

# 3. Re-run focused native tests

From a clean checkout at the branch tip:

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[mcp]'
python -m pip install pytest pytest-asyncio

python -m pytest -q \
  tests/test_dflash_sdpa256.py \
  tests/test_dflash_sdpa256_load_order.py \
  tests/test_dflash_sdpa256_lifecycle.py \
  tests/test_dflash_sdpa256_numerics.py \
  tests/test_sdpa256_attention.py \
  tests/test_dflash_prefill_memory_guard.py \
  tests/test_dflash_engine.py \
  tests/test_process_memory_enforcer.py
```

Record:

- exact branch SHA,
- Python version,
- MLX version,
- dflash-mlx revision/version,
- total pass/fail count.

Do not paste local paths or environment secrets into the public result.

---

# 4. Build the macOS application

Do **not** overwrite an existing installed oMLX application.

Build from the branch checkout:

```bash
apps/omlx-mac/Scripts/build.sh release
```

Expected staged application:

```text
apps/omlx-mac/build/Stage/oMLX.app
```

If the current machine normally uses the optional custom-kernel build, run a second build only after the plain release build succeeds:

```bash
apps/omlx-mac/Scripts/build.sh release --with-custom-kernel
```

Do not change multiple acceleration mechanisms at once merely to improve a benchmark score. The primary MTP-vs-DFlash comparison must use the **same application build** and change only the speculative decoder configuration.

Verify the staged application:

```bash
codesign --verify --deep --strict apps/omlx-mac/build/Stage/oMLX.app
```

Launch the staged app without replacing `/Applications/oMLX.app`.

Use an isolated test data root if practical. A generic pattern is:

```bash
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/omlx-dflash-ab.XXXXXX")"
OMLX_BASE_PATH="$TEST_ROOT" \
  apps/omlx-mac/build/Stage/oMLX.app/Contents/MacOS/oMLX
```

If model discovery requires an existing model directory, configure that directory locally in the isolated settings. Do not commit or publish its absolute path.

Before benchmarking, stop any other local oMLX server and avoid another heavy GPU/LLM workload.

---

# 5. Models

Target model for both configurations:

```text
Jundot/Qwen3.8-27B-oQ4e-mtp
```

DFlash draft:

```text
z-lab/Qwen3.8-27B-DFlash2
```

Record the exact local/Hugging Face revision used for each model.

Do not use a private model or private coding repository for this benchmark.

---

# 6. Fixed settings for the A/B

Keep these fixed for both configurations unless the branch/API rejects a combination:

```text
max_context_window:       131072
Memory Guard:             Balanced
Metal wired-memory limit: unchanged during the A/B
TurboQuant KV:            OFF
ANE prefill:              OFF
SpecPrefill:              OFF
VLM MTP:                  OFF
concurrent requests:      1
trust_remote_code:        OFF unless the target genuinely requires it
sampling temperature:     0 for synthetic benchmark
sampling top_p:           1
```

Do not tune the wired-memory ceiling between MTP and DFlash runs. Record the effective value locally; publishing the numeric limit is fine, but never publish privileged command history containing unrelated local information.

## Configuration A — Lightning MTP

```text
dflash_enabled:       false
mtp_enabled:          true
mtp_num_draft_tokens: 3
```

If `3` is no longer the current supported/default value, document what the branch actually resolves to. Do not silently tune MTP to make either side win.

## Configuration B — DFlash

```text
mtp_enabled:          false
dflash_enabled:       true
dflash_draft_model:   z-lab/Qwen3.8-27B-DFlash2
dflash_verify_mode:   adaptive (or branch default if unset)
dflash_max_ctx:       None / unlimited for the capacity test
```

For the decoder-isolation benchmark, disable DFlash's private prefix cache if possible so retained snapshots do not distort cold memory measurements:

```text
dflash_in_memory_cache: false
```

Then run a second daily-driver/cache phase with the private cache enabled under a deliberately bounded budget, for example:

```text
dflash_in_memory_cache:             true
dflash_in_memory_cache_max_entries: 1
dflash_in_memory_cache_max_bytes:   2 GiB
```

The 2 GiB / one-entry setting is an experimental starting point, not a claimed optimum. If a useful DFlash snapshot does not fit, record that fact rather than raising the budget immediately.

---

# 7. Benchmark hygiene

For every A/B pair:

1. Use the same application commit and same target checkpoint.
2. Run one inference request at a time.
3. Use the same prompt corpus and generation length.
4. Use the same sampling settings.
5. Return the system to a comparable idle state before each run.
6. Clear/restart caches consistently for cold measurements.
7. Do not compare a cold MTP request with a warm DFlash request.
8. Run at least three repetitions for the standard-size throughput tests.
9. Alternate order where practical (`MTP -> DFlash`, then `DFlash -> MTP`) to reduce thermal/order bias.
10. Record failures and guard aborts; do not discard failed trials from the final report.

Plug the machine into power and use the same macOS power mode throughout the comparison.

Do not intentionally drive the system into an OS-level OOM or kernel panic. Stop at the first sustained unsafe memory-pressure condition.

---

# 8. Standard throughput A/B

The existing oMLX benchmark backend can generate exact token-count code prompts and already records TTFT, processing throughput, generation throughput, cached tokens, and peak memory.

Use the `Code (Python)` benchmark corpus.

Standard prompt sizes:

```text
4K
8K
16K
32K
65K
```

Use:

```text
generation length: 256 tokens
batch size: 1
```

For each configuration and prompt length run at least three trials.

Collect:

```text
TTFT (ms)
prefill / processing tok/s
decode / generation tok/s
e2e latency (s)
peak MLX memory
process physical footprint / RSS high-water mark if available
cached tokens
DFlash acceptance ratio / accepted-from-draft if exposed
memory-guard throttle/rejection/abort events
swap delta
```

Also record median and worst-case result, not only the best run.

Do not publish full raw logs without sanitizing them first.

---

# 9. Long-context capacity test

The normal scheduler-based context benchmark cannot directly determine a DFlash admission boundary because DFlash bypasses the scheduler interface expected by that benchmark. Therefore run explicit staged real prefills.

Required stages:

```text
65K
80K
96K (target approximately 98K if the exact-token harness supports it safely)
```

Use short output first:

```text
max new tokens: 32
```

Only advance to the next stage if the previous one:

- completes cleanly,
- does not trigger repeated guard aborts,
- does not produce sustained heavy swap growth,
- leaves the app responsive,
- does not approach an obviously unsafe system-memory state.

For exact token counts, reuse `omlx.admin.benchmark._generate_prompt()` or the existing benchmark machinery rather than estimating from characters. Do not commit a one-off benchmark helper unless it is genuinely useful upstream.

At each stage record:

```text
prompt tokens
cold prefill time / tok/s
TTFT
decode tok/s
peak MLX active memory
peak process footprint
swap before / after
guard outcome
DFlash acceptance
success/failure
```

The main capacity question is:

> Can DFlash reliably complete approximately 96-98K active input on this 48 GiB machine with the normal memory guard enabled, without materially worse stability than MTP?

If MTP completes that level and DFlash consistently cannot, that is a material regression even if DFlash is faster at 16-32K.

---

# 10. Warm continuation / prefix-cache test

Cold synthetic throughput alone is not representative of an agent session.

For both MTP and the chosen DFlash cache configuration:

1. Send a long initial prompt (32K and 65K are sufficient for the first pass).
2. Allow the request to finish normally.
3. Append a small amount of new content without changing the prefix.
4. Run a second request.
5. Repeat once more.

Measure:

```text
cached tokens
follow-up TTFT
follow-up prefill tokens actually processed
decode tok/s
peak memory after each turn
snapshot/cache-store time if visible
cache hit kind if visible
```

Confirm that DFlash's private prefix cache does not create an unacceptable resident-memory increase or degrade continuation latency.

If DFlash requires a large RAM cache just to make continuations competitive, include that memory cost in the final decision.

---

# 11. Real coding-agent smoke test

Do not use a private work repository for a public benchmark report.

Create or use a disposable local toy repository containing only synthetic/public code. Example task:

```text
- inspect a small multi-file Python or Go project,
- find a deliberately inserted bug,
- modify two or more files,
- run tests,
- fix any failure,
- summarize the change.
```

Run the same task once with MTP and once with DFlash after a warm-up request.

If testing Qwen Code, configure its provider context to `131072` for this experiment so client auto-compaction does not artificially cap raw history around ~65K. The purpose is to determine whether the backend itself can sustain approximately 96-98K useful active history.

Record:

```text
total wall time
time waiting for model responses
number of model turns
tool-call correctness
any parser/tool errors
approximate context growth
whether compaction occurred
subjective latency only as a secondary note
```

Do not publish conversation content, repository paths, account identifiers, or tool output containing private machine information. Publish only aggregate timing/result data and a generic description of the synthetic task.

---

# 12. Decision criteria

Treat these as decision guidance, not hard scientific thresholds.

## Prefer DFlash as the daily driver if

All of the following are true:

- approximately **20-25% or greater median decode improvement** over MTP at the context sizes that matter for interactive coding (roughly 16K-65K),
- no severe cold-prefill/TTFT regression,
- stable approximately **90-98K raw active context** with the normal guard enabled,
- no repeated memory aborts or sustained swap thrashing,
- warm continuations/cache behavior is at least acceptable,
- coding-agent tool use remains correct,
- total task wall time improves materially rather than only raw decode TPS.

## Prefer MTP if

Any of the following are true:

- DFlash's median gain is only around 10-15%,
- DFlash reliably tops out around ~65K while MTP remains stable near ~96K,
- DFlash requires aggressive memory-limit changes to reach the same context,
- DFlash causes significant swap or UI/system responsiveness problems,
- continuation/cache behavior materially worsens real agent latency,
- tool/parser reliability regresses.

## Gray zone

A 15-25% speed improvement with a modest context reduction (for example 96K to ~80K) requires judgment. Report the data; do not automatically declare DFlash superior.

---

# 13. Required result report

Create a concise Markdown report locally. Before posting anything publicly, remove:

- home-directory user names,
- machine names,
- IP addresses,
- Tailnet details,
- API keys / auth headers,
- local account IDs,
- private source paths,
- private repository/project/client names,
- prompts or code copied from non-public work.

A public-safe report may contain:

- chip class (`M4 Pro`),
- unified-memory size (`48 GiB`),
- macOS major/minor version,
- oMLX commit SHA,
- MLX/dflash dependency versions,
- public model identifiers,
- benchmark settings,
- aggregate performance/memory metrics,
- sanitized error messages relevant to oMLX.

Use this table for the primary comparison:

| Context | Mode | Cold TTFT | Prefill tok/s | Decode tok/s | E2E | Peak MLX | Peak process | Swap delta | Cached tokens | Acceptance | Guard result |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4K | MTP | | | | | | | | | n/a | |
| 4K | DFlash | | | | | | | | | | |
| 8K | MTP | | | | | | | | | n/a | |
| 8K | DFlash | | | | | | | | | | |
| 16K | MTP | | | | | | | | | n/a | |
| 16K | DFlash | | | | | | | | | | |
| 32K | MTP | | | | | | | | | n/a | |
| 32K | DFlash | | | | | | | | | | |
| 65K | MTP | | | | | | | | | n/a | |
| 65K | DFlash | | | | | | | | | | |
| 80K | MTP | | | | | | | | | n/a | |
| 80K | DFlash | | | | | | | | | | |
| ~96-98K | MTP | | | | | | | | | n/a | |
| ~96-98K | DFlash | | | | | | | | | | |

Final recommendation must explicitly answer:

1. Is DFlash faster enough to matter in real use?
2. What is the largest stable raw active context for MTP and DFlash on this hardware?
3. Does DFlash materially increase resident/peak memory or swap?
4. Does warm continuation improve or worsen?
5. Does the coding-agent task complete faster end-to-end?
6. Should DFlash replace MTP as the default, remain optional, or be rejected for this machine?

Do not merge or mark the upstream PR ready solely because the benchmark is fast. Correctness review and long-context stability remain separate gates.
