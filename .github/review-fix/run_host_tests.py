"""Isolated host-only execution of exact repository routing/installer functions.

The checkout is complete, but native dependencies are stubbed. This is NOT a
full upstream pytest run, real DFlash lifecycle test, or numerical/Metal test.
"""
import ast
import logging
from pathlib import Path
import sys
import threading
from types import ModuleType
from typing import Optional

ROOT = Path.cwd()


def module(name, path=None):
    result = ModuleType(name)
    result.__package__ = name.rpartition('.')[0]
    if path is not None:
        result.__path__ = [str(path)]
        result.__package__ = name
    sys.modules[name] = result
    parent, _, attr = name.rpartition('.')
    if parent in sys.modules:
        setattr(sys.modules[parent], attr, result)
    return result


def selected_code(path, names=None, include_assignments=False):
    parsed = ast.parse(Path(path).read_text())
    nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    for node in parsed.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and (names is None or node.name in names):
            nodes.append(node)
        elif include_assignments and isinstance(node, (ast.Assign, ast.AnnAssign)):
            nodes.append(node)
    return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec')


def main():
    print('HOST-ONLY: exact repository functions; dependency/allocator/kernel stubs.', flush=True)
    import pytest
    import os
    import weakref
    from contextlib import contextmanager

    module('mlx', ROOT / '__stub_mlx')
    mx = module('mlx.core')
    mx.array = type('array', (), {})
    mx.get_active_memory = lambda: 0
    for package in ('mlx_lm', 'mlx_lm.models', 'mlx_vlm', 'mlx_vlm.models'):
        module(package, ROOT / '__stubs')
    for package in ('mlx_lm.models.base', 'mlx_vlm.models.base'):
        module(package).scaled_dot_product_attention = lambda *a, **kw: None
    module('omlx', ROOT / 'omlx')
    module('omlx.patches', ROOT / 'omlx/patches')
    monitor = module('omlx.memory_monitor')
    monitor.SDPA256_UNFUSED_SCORE_DTYPE_SIZE = 4
    exec(selected_code(ROOT / 'omlx/memory_monitor.py', {'estimate_unfused_sdpa_call_bytes'}), monitor.__dict__)
    monitor.register_tiled_prefill_head_dim = lambda *a, **kw: None
    sdpa = module('omlx.patches.sdpa256_attention')
    sdpa.__dict__.update(logging=logging, os=os, threading=threading, weakref=weakref,
                         contextmanager=contextmanager, mx=mx,
                         SDPA256_UNFUSED_SCORE_DTYPE_SIZE=4,
                         estimate_unfused_sdpa_call_bytes=monitor.estimate_unfused_sdpa_call_bytes)
    exec(selected_code(ROOT / 'omlx/patches/sdpa256_attention.py', include_assignments=True), sdpa.__dict__)
    tq = module('omlx.patches.turboquant_attention')
    tq.__dict__.update(_PATCHED=False, Optional=Optional, mx=mx,
                       logger=logging.getLogger('host-tq'),
                       _patch_update_eval_policy=lambda: None,
                       _patch_vlm_target_verify_attention=lambda: None)
    exec(selected_code(ROOT / 'omlx/patches/turboquant_attention.py', {'apply_turboquant_attention_patch'}), tq.__dict__)
    module('mlx_vlm.turboquant').TurboQuantKVCache = type('TurboQuantKVCache', (), {})
    tq_kv = module('omlx.turboquant_kv')
    tq_kv.BatchTurboQuantKVCache = type('BatchTurboQuantKVCache', (), {})
    tq_kv._state_length = lambda _: 0
    return pytest.main([
        'tests/test_dflash_sdpa256.py', 'tests/test_dflash_sdpa256_load_order.py',
        '--confcutdir=tests', '--noconftest', '-o', 'addopts=', '-q',
        '-k', 'not test_start_installs_route_before_loading_target and not test_grouped_dflash_prefill_reaches_bounded_route and not test_engine_event_factory_wraps_the_real_consumer_boundary',
    ])


if __name__ == '__main__':
    raise SystemExit(main())
