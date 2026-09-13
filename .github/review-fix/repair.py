"""Apply reviewed edits in an isolated checkout, validating every old blob first."""
from pathlib import Path
import ast
import hashlib

ORIGINALS = {
    'omlx/patches/sdpa256_attention.py': 'b8fadea97798a00e5bf2fa567bdb344a5a7b6e86',
    'omlx/patches/turboquant_attention.py': 'ca911a904c5a70b0fd4ba5f6e5c3d989e4a69608',
    'docs/dflash-sdpa256-validation.md': '8e3cb75f9c6d46a18fa04d3fc445b7f654c9ebb0',
}

RECONCILE = '''def _rebind_dflash_sdpa_aliases() -> None:
    """Repair aliases of known implementations, including pre-TurboQuant ones.

    Installation and alias coverage have different lifetimes. A DFlash helper
    can retain an original captured before another oMLX wrapper was installed,
    or can be imported/reloaded after our first installation. Follow only the
    explicit provenance attached by oMLX wrappers; never infer ownership from
    a function name or inspect arbitrary closures. Unrelated custom functions
    are left alone. This runs at engine load, not on the attention hot path.
    """
    import sys

    wrapper = _LM_SDPA256_WRAPPER
    if wrapper is None:
        return
    originals = []
    current = wrapper
    seen = {id(wrapper)}
    while True:
        # Read concrete metadata, not dynamic __getattr__ on callable objects.
        metadata = getattr(current, "__dict__", {})
        current = metadata.get("_omlx_sdpa_original")
        if current is None or id(current) in seen or not callable(current):
            break
        seen.add(id(current))
        originals.append(current)

    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("dflash_mlx."):
            continue
        implementation = getattr(module, "scaled_dot_product_attention", None)
        if any(implementation is original for original in originals):
            module.scaled_dot_product_attention = wrapper


'''


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Expected exactly one anchor: {old[:90]!r}')
    return text.replace(old, new, 1)


def transform_sdpa(text):
    text = replace_once(text, '_PATCHED = False\n', '_PATCHED = False\n_LM_SDPA256_WRAPPER = None\n')
    text = replace_once(text, 'def apply_sdpa256_attention_patch(', RECONCILE + 'def apply_sdpa256_attention_patch(')
    text = replace_once(text,
        '    global _PATCHED, _SDPA256_MIN_KV_LEN, _FORCE_TILED\n    if _PATCHED:\n        return False\n',
        '    global _PATCHED, _SDPA256_MIN_KV_LEN, _FORCE_TILED, _LM_SDPA256_WRAPPER\n'
        '    if _PATCHED:\n'
        '        _rebind_dflash_sdpa_aliases()\n'
        '        return False\n')
    text = replace_once(text,
        '    mlx_base.scaled_dot_product_attention = patched_sdpa\n',
        '    patched_sdpa._omlx_sdpa_original = original_sdpa\n'
        '    _LM_SDPA256_WRAPPER = patched_sdpa\n'
        '    mlx_base.scaled_dot_product_attention = patched_sdpa\n')
    text = replace_once(text,
        '    # Rebind model modules and DFlash helpers that captured the base\n'
        '    # function at import time (target_qwen_gdn and gqa_sdpa, #3241). Only\n',
        '    # Rebind model modules that captured the current base function.\n'
        '    # DFlash can hold an older pre-wrapper alias; reconcile it below. Only\n')
    text = replace_once(text,
        '        if mod is None or not mod_name.startswith(\n'
        '            ("mlx_lm.models.", "dflash_mlx.")\n'
        '        ):\n',
        '        if mod is None or not mod_name.startswith("mlx_lm.models."):\n')
    text = replace_once(text,
        '    # mlx-vlm carries its own base SDPA',
        '    _rebind_dflash_sdpa_aliases()\n\n'
        '    # mlx-vlm carries its own base SDPA')
    return text


def transform_turboquant(text):
    return replace_once(text,
        '    # Patch the module attribute\n'
        '    mlx_base.scaled_dot_product_attention = patched_sdpa\n',
        '    # Preserve concrete provenance so SDPA256 can recognize legitimate\n'
        '    # aliases captured by DFlash before this wrapper was installed.\n'
        '    patched_sdpa._omlx_sdpa_original = original_sdpa\n\n'
        '    # Patch the module attribute\n'
        '    mlx_base.scaled_dot_product_attention = patched_sdpa\n')

DOC_APPEND = '''

## Review follow-up: retained pre-wrapper aliases

SDPA256 now distinguishes base-wrapper installation from DFlash alias coverage.
TurboQuant records its original callable using private oMLX provenance metadata.
SDPA256 reconciles DFlash aliases against that identity chain on both first and
repeated installation. A legitimate reference captured before TurboQuant is
repaired; a genuinely custom function is not replaced. The base wrapper is not
stacked again, and a later outer wrapper on the base is left intact.

Run the additional regression tests alongside the suites above:

```bash
python -m pytest -q tests/test_dflash_sdpa256_load_order.py \\
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
'''


def main():
    changed = {}
    for name, expected in ORIGINALS.items():
        data = Path(name).read_bytes()
        actual = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        if actual != expected:
            raise RuntimeError(f'{name}: expected {expected}, found {actual}')
        text = data.decode()
        if name.endswith('/sdpa256_attention.py'):
            text = transform_sdpa(text)
        elif name.endswith('/turboquant_attention.py'):
            text = transform_turboquant(text)
        else:
            text += DOC_APPEND
        if name.endswith('.py'):
            ast.parse(text, filename=name)
        changed[name] = text
    for name, text in changed.items():
        Path(name).write_text(text)
    print('Applied three hash-verified edits')

if __name__ == '__main__':
    main()
