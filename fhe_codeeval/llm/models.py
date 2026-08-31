"""Convenience presets for API-hosted models used by FHE-CodeEval.

Open-weight models are commonly served through provider-specific
OpenAI-compatible endpoints. Configure those directly in a local YAML file
rather than committing endpoint details or credentials here.
"""

MODELS: dict[str, dict] = {
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-6",
        "max_tokens": 8192,
        "temperature": 0,
        "extra_params": {},
    },
    "gpt-5.4": {
        "provider": "openai",
        "model_id": "gpt-5.4",
        "max_tokens": 8192,
        "temperature": 0,
        "use_max_completion_tokens": True,
        "extra_params": {},
    },
}


def _normalise_provider(provider: str) -> str:
    provider = provider.lower().replace("_", "-")
    aliases = {
        "openai-compatible": "openai",
        "glm": "openai",
        "zhipu": "openai",
        "bigmodel": "openai",
        "anthropic-compatible": "anthropic",
        "claude": "anthropic",
    }
    return aliases.get(provider, provider)


def get_model_config(model_name: str, overrides: dict | None = None) -> dict:
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}

    if model_name in MODELS:
        config = {
            **MODELS[model_name],
            "extra_params": dict(MODELS[model_name].get("extra_params", {})),
        }
    elif overrides.get("provider"):
        config = {
            "provider": overrides["provider"],
            "model_id": model_name,
            "max_tokens": 8192,
            "temperature": 0,
            "extra_params": {},
        }
    else:
        raise ValueError(
            f"Unknown model {model_name!r}. Available presets: {list(MODELS)}. "
            "For a custom endpoint, set model_provider in the config."
        )

    extra_params = overrides.pop("extra_params", None)
    config.update(overrides)
    if extra_params:
        config["extra_params"] = {
            **config.get("extra_params", {}),
            **extra_params,
        }

    config["provider"] = _normalise_provider(config["provider"])
    return config
