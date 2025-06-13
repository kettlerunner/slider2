import unittest
import ast
import textwrap


def load_sanitize_function():
    with open('slider.py', 'r', encoding='utf-8') as f:
        source = f.read()
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'sanitize_text':
            func_code = textwrap.dedent(ast.get_source_segment(source, node))
            namespace = {'re': __import__('re')}
            exec(func_code, namespace)
            return namespace['sanitize_text']
    raise RuntimeError('sanitize_text not found')

sanitize_text = load_sanitize_function()

class TestSanitizeText(unittest.TestCase):
    def test_curly_quotes_and_dashes(self):
        raw = '“Hello” – it’s a test—indeed…'
        expected = '"Hello" - it\'s a test-indeed...'
        self.assertEqual(sanitize_text(raw), expected)


if __name__ == '__main__':
    unittest.main()
