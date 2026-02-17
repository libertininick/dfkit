# dfkit
A lightweight toolkit for connecting LangChain agents to dataframes, making it easy to ask natural language questions about your data.

## Quickstart

Coming soon!

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