# Examples

This directory contains examples demonstrating how to use `dfkit`.

## Prerequisites

1. Install `dfkit` (see the [main README](../README.md))
2. Create a `.env` file in the project root with your API key(s):
   ```
   ANTHROPIC_API_KEY=your-key-here
   ```

## [`config.py`](config.py)

Shared configuration used by examples. Provides:

- `APIKeys`: loads API keys from a `.env` file using `pydantic-settings`
- `ModelName`: enum of supported Claude models
- `get_chat_model()`: initializes a LangChain `BaseChatModel` with sensible defaults

## Examples

### [`diabetes_dataset_exploration.ipynb`](diabetes_dataset_exploration.ipynb)

End-to-end notebook showing how to use `dfkit` with a LangChain agent to analyze the scikit-learn diabetes dataset. Demonstrates:

- Loading data into a Polars DataFrame and registering it with `DataFrameToolkit`
- Providing column descriptions for richer agent context
- Extending the toolkit with `DecisionTreeModule` for interpretable analysis
- Creating a LangChain agent wired to the toolkit's tools and system prompt
- Asking natural language questions about the data (e.g., BMI vs. disease progression, gender differences, key risk factors)

### [`logging_example.py`](logging_example.py)

Shows how to enable and configure `dfkit`'s opt-in logging system. Demonstrates:

- Enabling logging with `enable_logging()` as a context manager
- Setting the log level (including the custom `TOOL_CALL` level)
- Choosing between `"short"` and `"full"` log formats
- Automatic error logging for failed operations
- Automatic cleanup when the context manager exits

Run it with:
```sh
uv run python examples/logging_example.py
```
