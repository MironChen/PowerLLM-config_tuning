import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


# Filled by configuration.Configuration when custom chat profiles are loaded.
_custom_chat_models: dict[str, dict[str, str]] = {}


def register_custom_chat_models(specs: dict[str, dict[str, str]]) -> None:
    """Replace the in-memory custom chat model specs (persisted in config.json)."""
    global _custom_chat_models
    _custom_chat_models = dict(specs)

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set")

MODEL_REGISTRY = {
    "gemini-3.1-flash-lite": {
        "display_name": "Google Gemini 3.1 Flash Lite",
        "provider": "google_genai",
        "model_name": "google_genai:gemini-3.1-flash-lite-preview",
    },
    "gemini-3-flash": {
        "display_name": "Google Gemini 3 Flash Preview",
        "provider": "google_genai",
        "model_name": "google_genai:gemini-3-flash-preview",
    },
    "gemini-3.1-pro": {
        "display_name": "Google Gemini 3.1 Pro Preview",
        "provider": "google_genai",
        "model_name": "google_genai:gemini-3.1-pro-preview",
    },
    "gemma-3n-e4b-googleAPI": {
        "display_name": "Gemma 3n E4B (Google API)",
        "provider": "google_genai",
        "model_name": "gemma-3n-e4b-it",
    },
    "gemma-3n-e4b": {
        "display_name": "Gemma 3n E4B",
        "provider": "openai_compatible",
        "model_name": "google/gemma-3n-e4b",
        "base_url": "http://127.0.0.1:1234/v1",
    },
    "gemma-3n-e4b-openrouter": {
        "display_name": "Gemma 3n E4B (OpenRouter)",
        "provider": "openai_compatible",
        "model_name": "google/gemma-3n-e4b-it:free",
        "base_url": "http://127.0.0.1:1234/v1",
    },
    
    "qwen-3.5-9b": {
        "display_name": "Qwen 3.5 9B",
        "provider": "openai_compatible",
        "model_name": "qwen/qwen3.5-9b",
        "base_url": "http://127.0.0.1:1234/v1",
    },
    "qwen-3-1.7b": {
        "display_name": "Qwen 3 1.7B",
        "provider": "openai_compatible",
        "model_name": "qwen/qwen3-1.7b",
        "base_url": "http://127.0.0.1:1234/v1",
    },
    "qwen-3.5-4b-mlx": {
        "display_name": "Qwen 3.5 4B MLX",
        "provider": "openai_compatible",
        "model_name": "Qwen3.5-4B-MLX-4bit",
        "base_url": "http://127.0.0.1:1234/v1",
    },
    "qwen-3.5-9b-openrouter": {
        "display_name": "Qwen 3.5 9B OpenRouter",
        "provider": "openai_compatible",
        "model_name": "qwen/qwen3.5-9b",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_API_KEY,
    },
}

def check_chat_model_connection(chat_model_id: str) -> bool:
    """
    Check if we can successfully connect to the chat model specified by chat_model_id.
    Returns True if connection is successful, False otherwise.
    """
    try:
        model = get_chat_model(chat_model_id)
        # Make a minimal chat request using the same message format used elsewhere in the project.
        model.invoke([("human", "Connection test.")])
        return True
    except Exception:
        return False

def normalize_chat_model_id(chat_model_id: str) -> str:
    if chat_model_id in MODEL_REGISTRY:
        return chat_model_id
    if chat_model_id in _custom_chat_models:
        return chat_model_id
    raise ValueError(f"Invalid chat model id: {chat_model_id}")


def get_model_spec(chat_model_id: str) -> dict[str, str]:
    if chat_model_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[chat_model_id]
    if chat_model_id in _custom_chat_models:
        spec = _custom_chat_models[chat_model_id]
        return {
            "display_name": spec["display_name"],
            "provider": "openai_compatible",
            "model_name": spec["model_name"],
            "base_url": spec["base_url"],
            "api_key": spec.get("api_key", ""),
        }
    raise ValueError(f"Invalid chat model id: {chat_model_id}")


def get_available_llms():
    return [
        {"id": model_id, "name": spec["display_name"]}
        for model_id, spec in MODEL_REGISTRY.items()
    ]


def resolve_chat_model_name(chat_model_id: str) -> str:
    return get_model_spec(chat_model_id)["model_name"]


def _openai_compatible_api_key(spec: dict[str, str]) -> SecretStr:
    key = (spec.get("api_key") or "").strip()
    if key:
        return SecretStr(key)
    # Some local servers accept any placeholder when auth is disabled.
    return SecretStr("Bearer omlx")


def get_chat_model(chat_model_id: str):
    """
    Return a configured chat model instance for a registered or custom model ID.
    """
    load_dotenv()
    spec = get_model_spec(chat_model_id)
    provider = spec["provider"]

    if provider == "google_genai":
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("Please set GOOGLE_API_KEY in your .env file")
        os.environ["GOOGLE_API_KEY"] = google_api_key
        model_name = spec["model_name"]
        provider_prefix = f"{provider}:"
        # Normalize ids like "google_genai:gemini-..." to plain provider model ids.
        if model_name.startswith(provider_prefix):
            model_name = model_name[len(provider_prefix) :]
        return init_chat_model(model_name, model_provider=provider)

    if provider == "openai_compatible":
        return ChatOpenAI(
            model=spec["model_name"],
            api_key=_openai_compatible_api_key(spec),
            base_url=spec["base_url"],
            reasoning={"effort": "none"},
        )

    raise ValueError(f"Unsupported provider: {provider}")
