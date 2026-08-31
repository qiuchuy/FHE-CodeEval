"""
LLM API client for FHE benchmark prompt-response generation.

Supports Anthropic and OpenAI providers. Generation parameters come from the
selected model preset and the local run config.

Usage:
    from llm.client import generate_fhe_kernel
    code = generate_fhe_kernel(prompt, model_name="claude-sonnet-4-6")
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

from fhe_codeeval.llm.models import get_model_config


def _extract_code(text: str) -> str:
    """
    Strip markdown code fences from an LLM response if present.
    Returns the raw Python source.
    """
    # Match ```python ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return text.strip()


ChatMessage = dict[str, str]
_DEFAULT_HTTP_TIMEOUT_SECONDS = 1800.0


def _configured_secret(config: dict, key_name: str, env_name_key: str, default_env: str) -> str | None:
    explicit = config.get(key_name)
    if explicit:
        return explicit
    env_name = config.get(env_name_key) or default_env
    return os.environ.get(env_name)


def _http_timeout_seconds(config: dict) -> float:
    raw = config.get("timeout_seconds", config.get("timeout"))
    if raw is None:
        return _DEFAULT_HTTP_TIMEOUT_SECONDS
    try:
        timeout = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout_seconds must be a positive number") from exc
    if timeout <= 0:
        raise ValueError("timeout_seconds must be a positive number")
    return timeout


def _configured_retries(config_overrides: dict | None, default: int) -> int:
    raw = (config_overrides or {}).get("retries")
    if raw is None:
        return default
    try:
        retries = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("retries must be a non-negative integer") from exc
    if retries < 0:
        raise ValueError("retries must be a non-negative integer")
    return retries


UsageDict = dict[str, int]


def _empty_usage() -> UsageDict:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def _merge_usage(acc: UsageDict, new: UsageDict) -> UsageDict:
    result = dict(acc)
    for k, v in new.items():
        if isinstance(v, (int, float)):
            result[k] = result.get(k, 0) + int(v)
    return result


def _call_anthropic_messages(messages: list[ChatMessage], config: dict) -> tuple[str, UsageDict]:
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed. Run: uv sync --locked")

    api_key = _configured_secret(config, "api_key", "api_key_env", "ANTHROPIC_API_KEY")
    if not api_key:
        env_name = config.get("api_key_env") or "ANTHROPIC_API_KEY"
        raise EnvironmentError(f"{env_name} environment variable not set")

    base_url = _configured_secret(config, "base_url", "base_url_env", "ANTHROPIC_BASE_URL")
    client_kwargs = {
        "api_key": api_key,
        "timeout": _http_timeout_seconds(config),
    }
    if base_url:
        client_kwargs["base_url"] = base_url
    client = anthropic.Anthropic(**client_kwargs)

    extra_params = dict(config.get("extra_params", {}))
    prompt_caching = bool(extra_params.pop("prompt_caching", False))

    # Anthropic API uses a dedicated `system` param; extract system-role messages.
    system_texts = [m["content"] for m in messages if m.get("role") == "system"]
    chat_messages = [m for m in messages if m.get("role") != "system"]

    create_kwargs: dict = {
        "model": config["model_id"],
        "max_tokens": config["max_tokens"],
        "temperature": config["temperature"],
    }
    protected = {"model", "messages", "system", "max_tokens", "temperature"}
    create_kwargs.update({key: value for key, value in extra_params.items() if key not in protected})

    if system_texts:
        system_text = "\n\n".join(system_texts)
        if prompt_caching:
            create_kwargs["system"] = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]
        else:
            create_kwargs["system"] = system_text

    if prompt_caching:
        # Mark the last human turn so the full prefix up to that point is cached.
        processed: list[ChatMessage] = []
        for i, msg in enumerate(chat_messages):
            if i == len(chat_messages) - 1 and msg.get("role") == "user":
                content = msg["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
                processed.append({**msg, "content": content})
            else:
                processed.append(msg)
        create_kwargs["messages"] = processed
    else:
        create_kwargs["messages"] = chat_messages

    message = client.messages.create(**create_kwargs)
    u = message.usage
    usage: UsageDict = {
        "input_tokens": getattr(u, "input_tokens", 0) or 0,
        "output_tokens": getattr(u, "output_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
    }
    return message.content[0].text, usage


def _normalize_openai_base_url(base_url: str | None) -> str | None:
    """Strip a trailing /chat/completions so the SDK does not request …/chat/completions/chat/completions."""
    if not base_url:
        return base_url
    u = base_url.strip().rstrip("/")
    if u.endswith("/chat/completions"):
        u = u[: -len("/chat/completions")].rstrip("/")
    return u


def _openai_request_kwargs(messages: list[ChatMessage], config: dict) -> dict:
    """Build OpenAI chat.completions kwargs, including provider-specific extra_body."""
    extra_params = dict(config.get("extra_params", {}))
    raw_extra_body = extra_params.pop("extra_body", None)
    extra_body = dict(raw_extra_body) if isinstance(raw_extra_body, dict) else {}

    if "enable_thinking" in config:
        extra_body["enable_thinking"] = config["enable_thinking"]

    use_max_completion_tokens = config.get("use_max_completion_tokens", False)
    kwargs: dict = {
        "model": config["model_id"],
        "messages": messages,
        **extra_params,
    }
    if use_max_completion_tokens:
        kwargs["max_completion_tokens"] = config["max_tokens"]
    else:
        kwargs["max_tokens"] = config["max_tokens"]
    # temperature is not supported for o-series models
    if config["temperature"] != 1 or not config["model_id"].startswith("o"):
        kwargs["temperature"] = config["temperature"]
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def _call_openai_messages(messages: list[ChatMessage], config: dict) -> tuple[str, UsageDict]:
    try:
        import openai
    except ImportError:
        raise ImportError("openai package not installed. Run: uv sync --locked")

    api_key = _configured_secret(config, "api_key", "api_key_env", "OPENAI_API_KEY")
    if not api_key:
        env_name = config.get("api_key_env") or "OPENAI_API_KEY"
        raise EnvironmentError(f"{env_name} environment variable not set")

    base_url = _normalize_openai_base_url(_configured_secret(config, "base_url", "base_url_env", "OPENAI_BASE_URL"))
    client_kwargs = {
        "api_key": api_key,
        "timeout": _http_timeout_seconds(config),
    }
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.OpenAI(**client_kwargs)
    kwargs = _openai_request_kwargs(messages, config)
    response = client.chat.completions.create(**kwargs)
    u = response.usage
    usage: UsageDict = {
        "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(u, "completion_tokens", 0) or 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    return response.choices[0].message.content, usage


def call_llm_messages(
    messages: list[ChatMessage],
    model_name: str,
    config_overrides: dict | None = None,
) -> tuple[str, UsageDict]:
    """
    Send chat messages to the specified LLM and return (response_text, usage).

    Args:
        messages:    Chat-completion messages.
        model_name:  Key in llm.models.MODELS.
    Returns:
        (raw response text, token usage dict)
    """
    config = get_model_config(model_name, overrides=config_overrides)
    provider = config["provider"]

    if provider == "anthropic":
        return _call_anthropic_messages(messages, config)
    elif provider == "openai":
        return _call_openai_messages(messages, config)
    else:
        raise ValueError(f"Unknown provider '{provider}'")


def call_llm(
    prompt: str,
    model_name: str,
    config_overrides: dict | None = None,
) -> tuple[str, UsageDict]:
    """
    Send a single user prompt to the specified LLM and return (response_text, usage).
    """
    return call_llm_messages(
        [{"role": "user", "content": prompt}],
        model_name,
        config_overrides=config_overrides,
    )


def generate_fhe_kernel_from_messages(
    messages: list[ChatMessage],
    model_name: str,
    output_path: Optional[str] = None,
    retries: int = 2,
    config_overrides: dict | None = None,
) -> tuple[str, UsageDict]:
    """
    Call the LLM with chat messages, extract Python code from the response,
    optionally save to `output_path`, and return (code, cumulative_usage).
    """
    retry_count = _configured_retries(config_overrides, retries)
    last_exc: Optional[Exception] = None
    cumulative_usage = _empty_usage()
    for attempt in range(retry_count + 1):
        try:
            raw, usage = call_llm_messages(messages, model_name, config_overrides=config_overrides)
            cumulative_usage = _merge_usage(cumulative_usage, usage)
            code = _extract_code(raw)
            if output_path:
                from pathlib import Path

                Path(output_path).write_text(code)
            return code, cumulative_usage
        except Exception as exc:
            last_exc = exc
            if attempt < retry_count:
                time.sleep(2**attempt)  # brief backoff
    raise RuntimeError(f"LLM call failed after {retry_count + 1} attempts: {last_exc}") from last_exc


def generate_fhe_kernel(
    prompt: str,
    model_name: str,
    output_path: Optional[str] = None,
    retries: int = 2,
    config_overrides: dict | None = None,
) -> tuple[str, UsageDict]:
    """
    Call the LLM with `prompt`, extract Python code from the response,
    optionally save to `output_path`, and return (code, cumulative_usage).

    Args:
        prompt:       Full prompt text (system + task description + source).
        model_name:   Model key from llm.models.MODELS.
        output_path:  If set, write the extracted code to this file path.
        retries:      Number of retry attempts on transient API errors.
    Returns:
        (Python source code string, token usage dict)
    """
    messages = [{"role": "user", "content": prompt}]
    return generate_fhe_kernel_from_messages(
        messages,
        model_name,
        output_path=output_path,
        retries=retries,
        config_overrides=config_overrides,
    )
