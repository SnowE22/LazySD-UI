"""Regression: api_generate must not re-import module-level names in nested blocks.

A conditional/try-nested ``import uuid`` made ``uuid`` a function-local name for
the WHOLE function, so ``uuid.uuid4()`` raised UnboundLocalError on every
generation request whenever the branch was not taken.
"""
import ast
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app.py"


def test_api_generate_does_not_shadow_module_imports():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    module_imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_imported |= {(a.asname or a.name).split(".")[0] for a in node.names}
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "api_generate"
    )
    top_level = {id(n) for n in fn.body}
    bad = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Import) and id(node) not in top_level:
            for a in node.names:
                name = (a.asname or a.name).split(".")[0]
                if name in module_imported:
                    bad.append((name, node.lineno))
    assert not bad, f"nested local re-import shadows module name (UnboundLocalError): {bad}"
