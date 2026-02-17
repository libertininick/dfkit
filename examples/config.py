"""Tools for API key management and LLM configuration."""

from enum import StrEnum

from langchain.chat_models import BaseChatModel, init_chat_model
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings


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

    Notes:
        - Anthropic models names: https://docs.claude.com/en/docs/about-claude/models/overview
    """

    CLAUDE_HAIKU = "claude-haiku-4-5"
    CLAUDE_SONNET = "claude-sonnet-4-5"
    CLAUDE_OPUS = "claude-opus-4-6"


def get_chat_model(
    *,
    model_name: ModelName = ModelName.CLAUDE_HAIKU,
    timeout: int | None = 60,
    max_retries: int = 2,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> BaseChatModel:
    """Initialize a chat model based on the specified model name and API keys.

    Args:
        model_name (ModelName): The name of the model to initialize. Defaults to ModelName.CLAUDE_HAIKU.
        timeout (int | None, optional): Timeout for API requests in seconds. Defaults to 60 seconds.
        max_retries (int, optional): Maximum number of retries for API requests in case of failures. Defaults to 2.
        temperature (float, optional): Sampling temperature for the model. Defaults to 0.0.
        max_tokens (int, optional): Maximum number of tokens for the model's response. Defaults to 4096.

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
        case _:
            raise ValueError(f"Unsupported model name: {model_name}")

    # Initialize a chat model
    chat_model = init_chat_model(
        model=model_name.value,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return chat_model
