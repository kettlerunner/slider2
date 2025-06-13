import ast
import re
from pathlib import Path


def load_sanitize_text():
    slider_path = Path(__file__).resolve().parents[1] / "slider.py"
    source = slider_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "sanitize_text":
            code = ast.get_source_segment(source, node)
            ns = {"re": re}
            exec(code, ns)
            return ns["sanitize_text"]
    raise RuntimeError("sanitize_text not found")


sanitize_text = load_sanitize_text()

def test_curly_quotes_and_dashes():
    text = '“Hello” – “World” — test'
    result = sanitize_text(text)
    assert result == '"Hello" - "World" - test'
    assert all(ord(c) < 128 for c in result)


def test_single_quotes_and_dashes():
    text = "‘Foo’–bar—baz"
    result = sanitize_text(text)
    assert result == "'Foo'-bar-baz"
    assert all(ord(c) < 128 for c in result)