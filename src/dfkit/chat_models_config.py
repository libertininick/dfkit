"""Tools for API key management and LLM configuration shared across examples, evals, and notebooks."""

from enum import StrEnum
from typing import Any, Final

from langchain.chat_models import BaseChatModel, init_chat_model
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings

CLAUDE_VERSION: Final[str] = "4-6"
GPT_VERSION: Final[str] = "5.4"


class APIKeys(BaseSettings, env_file=".env", env_file_encoding="utf-8", extra="ignore"):
    """API keys for various LLM providers and services.

    Attributes:
        anthropic (SecretStr): API key for Anthropic LLMs.
        openai (SecretStr): API key for OpenAI LLMs.
    """

    anthropic: SecretStr = Field(default=SecretStr(""), description="API key for Anthropic LLMs.")
    openai: SecretStr = Field(default=SecretStr(""), description="API key for OpenAI LLMs.")


class ModelName(StrEnum):
    """Enumeration of supported LLM model names.

    Attributes:
        CLAUDE_HAIKU: Anthropic Claude Haiku model.
        CLAUDE_SONNET: Anthropic Claude Sonnet model.
        CLAUDE_OPUS: Anthropic Claude Opus model.
        GPT_NANO: OpenAI GPT Nano model.
        GPT_MINI: OpenAI GPT Mini model.
        GPT: OpenAI GPT model.

    Notes:
        - Anthropic models: https://docs.claude.com/en/docs/about-claude/models/overview
        - OpenAI models: https://developers.openai.com/api/docs/models
    """

    CLAUDE_HAIKU = "claude-haiku-4-5"  # No 4-6 model yet
    CLAUDE_SONNET = f"claude-sonnet-{CLAUDE_VERSION}"
    CLAUDE_OPUS = f"claude-opus-{CLAUDE_VERSION}"
    GPT_NANO = f"gpt-{GPT_VERSION}-nano"
    GPT_MINI = f"gpt-{GPT_VERSION}-mini"
    GPT = f"gpt-{GPT_VERSION}"


def get_chat_model(
    *,
    model_name: ModelName = ModelName.GPT_MINI,
    timeout: int | None = 60,
    max_retries: int = 2,
    temperature: float = 0.0,
    **kwargs: Any,
) -> BaseChatModel:
    """Initialize a chat model based on the specified model name and API keys.

    Args:
        model_name (ModelName): The name of the model to initialize. Defaults to ModelName.GPT_MINI.
        timeout (int | None, optional): Timeout for API requests in seconds. Defaults to 60 seconds.
        max_retries (int, optional): Maximum number of retries for API requests in case of failures. Defaults to 2.
        temperature (float, optional): Sampling temperature for the model. Defaults to 0.0.
        **kwargs (Any): Additional keyword arguments to pass to `init_chat_model`.

    Returns:
        BaseChatModel: A chat model instance initialized with the specified parameters.

    Raises:
        ValueError: If an unsupported model name is provided.

    """
    # Load API keys from .env file
    api_keys = APIKeys()

    # Get API key based on the model name
    match model_name:
        case ModelName.CLAUDE_HAIKU | ModelName.CLAUDE_SONNET | ModelName.CLAUDE_OPUS:
            api_key = api_keys.anthropic
        case ModelName.GPT_NANO | ModelName.GPT_MINI | ModelName.GPT:
            api_key = api_keys.openai
        case _:
            raise ValueError(f"Unsupported model name: {model_name}")

    # Initialize a chat model
    chat_model = init_chat_model(
        model=model_name.value,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        temperature=temperature,
        **kwargs,
    )

    return chat_model
