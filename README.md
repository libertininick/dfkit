# dfkit
A lightweight toolkit for connecting LangChain agents to dataframes, making it easy to ask natural language questions about your data.

## Motivation

Tabular data — especially when it spans thousands of rows or lives across multiple tables — can quickly consume an LLM's entire context window. Yet these models excel at reasoning about and analyzing structured data when they can access it effectively. `dfkit` bridges that gap by equipping LLM agents with tools to query and interact with dataframes directly, powering rich analysis without ever pulling the raw data into the context window.

## Quickstart

### 1. Install dfkit

```sh
uv pip install git+https://github.com/libertininick/dfkit.git
```

### 2. Set up your data and toolkit

```python
import polars as pl
from dfkit import DataFrameToolkit

# Load your data into a Polars DataFrame
df = pl.DataFrame({
    "product": ["Widget A", "Widget B", "Widget C"],
    "revenue": [1200, 3400, 5600],
    "units_sold": [120, 340, 560],
})

# Initialize the toolkit and register your DataFrame
toolkit = DataFrameToolkit()
toolkit.register_dataframe(
    name="Sales Data",
    dataframe=df,
    description="Quarterly product sales with revenue and units sold.",
    column_descriptions={
        "product": "Product name.",
        "revenue": "Total revenue in USD.",
        "units_sold": "Number of units sold.",
    },
)
```

### 3. Create an agent and ask questions

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

# Initialize your LLM (requires an API key set in your environment)
model = init_chat_model("claude-haiku-4-5")

# Create an agent with the toolkit's tools and system prompt
agent = create_agent(
    model=model,
    tools=toolkit.get_tools(),
    system_prompt=toolkit.get_system_prompt(),
)

# Ask a question about your data
response = agent.invoke({
    "messages": [HumanMessage("Which product has the highest revenue?")]
})
print(response["messages"][-1].content)
```

The agent uses the toolkit to translate your natural language questions into
SQL queries against the registered DataFrames and returns the results.

## Development Setup

Using `uv` for Python environment management:
```sh
# Clone repository
git clone git@github.com:libertininick/dfkit.git
cd dfkit

# Install dependencies
uv sync

# Set up pre-commit hooks
uv run pre-commit install
```

### Updating dependencies

#### Update a single dependency

```sh
uv lock --upgrade-package <package name>
uv pip show <package name>
```

#### Update uv tool and all dependencies

1. Update `uv` tool
2. Upgrade `Python` version installed by `uv`
3. Upgrade all dependencies in `uv.lock` file
4. Sync virtual environment with updated dependencies
5. Prune `uv` cache to remove dependencies that are no longer needed

```sh
uv self update \
&& uv python upgrade \
&& uv lock --upgrade \
&& uv sync \
&& uv cache prune
```

### Update pre-commit hooks:

```sh
uv run pre-commit install-hooks \
&& uv run pre-commit autoupdate
```

### Run checks

```sh
uv run ruff check . \
&& uv run ty check . \
&& uv tool run pydoclint src/ tests/ \
&& uv run pytest --cov src/ .
```

---