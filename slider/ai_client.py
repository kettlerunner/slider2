"""OpenAI client wrapper. All model calls in the app go through this module.

The Responses API is the primary path: it supports the web_search tool,
structured JSON output (json_schema), and reasoning-effort control on newer
models. Chat Completions is the fallback for models that reject Responses.
The client is created lazily so importing this module never touches the network.
"""

from slider import config

try:
    from openai import OpenAI
except Exception:  # SDK missing or broken install
    OpenAI = None

_client = None
_client_failed = False


def _get_client():
    global _client, _client_failed
    if _client is not None or _client_failed:
        return _client
    if OpenAI is None or not config.OPENAI_API_KEY:
        _client_failed = True
        return None
    try:
        kwargs = {
            "api_key": config.OPENAI_API_KEY,
            "timeout": float(config.OPENAI_REQUEST_TIMEOUT),
            "max_retries": 1,
        }
        if config.OPENAI_API_BASE:
            kwargs["base_url"] = config.OPENAI_API_BASE
        _client = OpenAI(**kwargs)
    except Exception as exc:
        print(f"Failed to initialize OpenAI client: {exc}")
        _client_failed = True
    return _client


def is_available() -> bool:
    """True when the SDK is installed and an API key is configured."""
    return OpenAI is not None and bool(config.OPENAI_API_KEY)


def _clean(text):
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def _supports_reasoning(model: str) -> bool:
    name = (model or "").lower()
    return name.startswith(("gpt-5", "o1", "o3", "o4"))


def _build_kwargs(prompt, model, tool_type, json_schema, schema_name, with_reasoning):
    kwargs = {"model": model, "input": prompt}
    if tool_type:
        kwargs["tools"] = [{"type": tool_type}]
    if json_schema:
        kwargs["text"] = {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": json_schema,
                "strict": True,
            }
        }
    if with_reasoning:
        kwargs["reasoning"] = {"effort": config.OPENAI_REASONING_EFFORT}
    return kwargs


def responses_create(
    prompt: str,
    model: str = None,
    use_web_search: bool = False,
    json_schema: dict = None,
    schema_name: str = "result",
):
    """Call the Responses API. Returns the output text or None on any failure.

    Optional features degrade gracefully: if the model rejects the reasoning
    parameter, the structured-output format, or the web_search tool name, the
    call is retried without that feature before giving up.
    """
    client = _get_client()
    if client is None:
        return None
    model = model or config.OPENAI_CHAT_MODEL

    tool_types = ["web_search", "web_search_preview"] if use_web_search else [None]
    with_reasoning = _supports_reasoning(model) and bool(config.OPENAI_REASONING_EFFORT)
    schema = json_schema
    last_error = None

    for tool_type in tool_types:
        for _ in range(3):  # at most: drop reasoning, then drop schema
            kwargs = _build_kwargs(prompt, model, tool_type, schema, schema_name, with_reasoning)
            try:
                response = client.responses.create(**kwargs)
                return _clean(getattr(response, "output_text", None))
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                if with_reasoning and "reasoning" in message:
                    with_reasoning = False
                    continue
                if schema and ("json_schema" in message or "text.format" in message or "format" in message):
                    schema = None
                    continue
                break
        if not (tool_type and "web_search" in str(last_error).lower()):
            break

    print(f"OpenAI responses request failed: {last_error}")
    if not use_web_search:
        return chat_completion(prompt, model=model)
    return None


def chat_completion(prompt: str, model: str = None):
    """Call Chat Completions. Returns the reply text or None on any failure."""
    client = _get_client()
    if client is None:
        return None
    model = model or config.OPENAI_CHAT_MODEL
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return _clean(response.choices[0].message.content)
    except Exception as exc:
        print(f"OpenAI chat completion failed: {exc}")
        return None
