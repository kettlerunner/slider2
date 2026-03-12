"""OpenAI client initialization and wrapper functions.

Uses the Responses API (recommended) as the primary path,
with Chat Completions as a lightweight alternative.
"""

from slider import config

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

_client = None

if OpenAI is not None and config.OPENAI_API_KEY:
    try:
        _client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            timeout=float(config.OPENAI_REQUEST_TIMEOUT),
        )
    except Exception as exc:
        print(f"Failed to initialize OpenAI client: {exc}")


def is_available() -> bool:
    """Return True if the OpenAI client is initialized and ready."""
    return _client is not None


def responses_create(
    prompt: str,
    model: str = None,
    use_web_search: bool = False,
) -> str | None:
    """Call OpenAI Responses API. Returns response text or None.

    Args:
        prompt: The prompt text to send.
        model: Model name (defaults to config.OPENAI_CHAT_MODEL).
        use_web_search: If True, enables the web_search tool.
    """
    if _client is None:
        return None

    model = model or config.OPENAI_CHAT_MODEL

    kwargs = {"model": model, "input": prompt}
    if use_web_search:
        kwargs["tools"] = [{"type": "web_search"}]

    try:
        response = _client.responses.create(**kwargs)
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        return None
    except Exception as exc:
        print(f"OpenAI responses request failed: {exc}")
        return None


def chat_completion(
    prompt: str,
    model: str = None,
) -> str | None:
    """Call OpenAI Chat Completions API. Returns response text or None.

    Args:
        prompt: The prompt text to send.
        model: Model name (defaults to config.OPENAI_CHAT_MODEL).
    """
    if _client is None:
        return None

    model = model or config.OPENAI_CHAT_MODEL

    try:
        response = _client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content
        if isinstance(text, str) and text.strip():
            return text.strip()
        return None
    except Exception as exc:
        print(f"OpenAI chat completion failed: {exc}")
        return None
