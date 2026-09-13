"""Hash-checked FA256 provenance and test isolation follow-up."""
import ast
import hashlib
from pathlib import Path
from repair import replace_once

ORIGINALS = {
    'omlx/patches/qwen35_fa256_attention.py': '5ef9352e4b09c9599e7916241bcb080b18cea0c7',
    'tests/test_dflash_sdpa256.py': '3ad75eb447eaf75b8319de63cc63562ed06ae76f',
    'tests/test_dflash_sdpa256_load_order.py': '321c9a00371b1fcf1d1a5809beea6251d4cffedc',
}
FA_TESTS = '''

@pytest.mark.parametrize("order", ["fa", "tq_fa", "fa_tq"])
def test_fa256_provenance_survives_supported_wrapper_chains(runtime, monkeypatch, order):
    from omlx.patches import qwen35_fa256_attention as fa

    monkeypatch.setattr(fa, "_PATCHED", False)
    monkeypatch.setenv("OMLX_FA256_STEEL", "1")
    monkeypatch.setenv("OMLX_FA256_DISPATCH_BUDGET", "0")
    monkeypatch.setattr(fa, "_native_kernel", lambda: Mock())
    monkeypatch.setattr(fa._fa256_fast, "fa256_supports_dispatch_budget", lambda: True)
    # Installer integration, not a steel-kernel benchmark. The FA wrapper
    # delegates to its prior function for this simulated unsupported shape.
    monkeypatch.setattr(fa, "_should_route", lambda *args: False)
    if order == "tq_fa":
        assert tq.apply_turboquant_attention_patch()
    assert fa.apply_qwen35_fa256_attention_patch()
    if order == "fa_tq":
        assert tq.apply_turboquant_attention_patch()
    assert sdpa.apply_sdpa256_attention_patch()
    assert_bounded(runtime)
    installed = runtime.base.scaled_dot_product_attention
    for module in runtime.modules:
        module.scaled_dot_product_attention = runtime.original
    assert sdpa.apply_sdpa256_attention_patch() is False
    assert runtime.base.scaled_dot_product_attention is installed
    assert_bounded(runtime)
'''


def main():
    changed = {}
    for name, expected in ORIGINALS.items():
        data = Path(name).read_bytes()
        actual = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        if actual != expected:
            raise RuntimeError(f'{name}: expected {expected}, found {actual}')
        text = data.decode()
        if name.endswith('qwen35_fa256_attention.py'):
            text = replace_once(text,
                '        mlx_base.scaled_dot_product_attention = patched_lm_sdpa\n',
                '        # SDPA256 must recognize DFlash aliases captured before this wrap.\n'
                '        patched_lm_sdpa._omlx_sdpa_original = original_lm_sdpa\n'
                '        mlx_base.scaled_dot_product_attention = patched_lm_sdpa\n')
        elif name.endswith('test_dflash_sdpa256.py'):
            text = replace_once(text,
                'def isolated_provider(monkeypatch):\n',
                'def isolated_provider(monkeypatch):\n'
                '    monkeypatch.setattr(sdpa, "_LM_SDPA256_WRAPPER", sdpa._LM_SDPA256_WRAPPER)\n')
        else:
            text += FA_TESTS
        ast.parse(text, filename=name)
        changed[name] = text
    for name, text in changed.items():
        Path(name).write_text(text)
    doc = Path('docs/dflash-sdpa256-validation.md')
    text = doc.read_text().replace(
        'TurboQuant records its original callable using private oMLX provenance metadata.',
        'TurboQuant and FA256 record their original callables using private oMLX provenance metadata.')
    doc.write_text(text)
    print('Applied FA256 provenance, regression cases, and original-fixture isolation')

if __name__ == '__main__':
    main()
