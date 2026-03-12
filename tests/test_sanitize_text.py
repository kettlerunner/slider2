from slider.utils import sanitize_text


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