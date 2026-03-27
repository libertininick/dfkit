"""Decision tree tool module for LLM agents."""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool
from loguru import logger

from dfkit.tool_modules.decision_tree.fitting import AUTO_MIN_SAMPLES_FLOOR, MAX_TREE_DEPTH, MIN_TREE_DEPTH
from dfkit.tool_modules.decision_tree.fitting import analyze_with_decision_tree as _analyze_with_decision_tree
from dfkit.tool_modules.decision_tree.models import DecisionTreeResult
from dfkit.tool_modules.models import ToolCallError
from dfkit.tool_modules.tool_module_context import ToolModuleContext
from dfkit.utils.logging import TOOL_CALL_LEVEL

# region Public interface


class DecisionTreeModule:
    """ToolModule that provides an analyze_with_decision_tree tool for LLM agents.

    Wraps `analyze_with_decision_tree` as a LangChain tool, converting
    `ValueError` exceptions into `ToolCallError` responses so the LLM agent
    can self-correct on invalid inputs.

    Examples:
        >>> from dfkit import DataFrameToolkit
        >>> from dfkit.tool_modules.decision_tree import DecisionTreeModule

        Initialize toolkit
        >>> toolkit = DataFrameToolkit()

        Get core tools + decision tree tools
        >>> tools = toolkit.get_tools(DecisionTreeModule)
        >>> assert "analyze_with_decision_tree" in {t.name for t in tools}

        Get core prompt + decision tree prompt
        >>> system_prompt = toolkit.get_system_prompt(DecisionTreeModule)
    """

    def __init__(self, context: ToolModuleContext) -> None:
        """Initialize DecisionTreeModule.

        Args:
            context (ToolModuleContext): The tool module context used to access
                registered DataFrames.
        """
        self._context = context
        self._tools = (tool(self.analyze_with_decision_tree),)

    @property
    def system_prompt(self) -> str:
        """LLM guidance for using this module's tools.

        Returns:
            str: System prompt text explaining when and how to use the module's tools.
        """
        return (
            "You have access to an **analyze_with_decision_tree** tool that discovers "
            "relationships between columns in a DataFrame by fitting a decision tree.\n\n"
            "**When to use it**: Use this tool for insight discovery - to understand "
            "which features best predict or segment a target column. This is an "
            "exploratory analysis tool, not a production prediction model.\n\n"
            "**Parameters**:\n"
            "- `dataframe`: Name or ID of a registered DataFrame.\n"
            "- `target`: The column you want to predict or segment (e.g., 'churn', 'revenue').\n"
            "- `features`: Optional list of feature columns. Omit to use all suitable columns.\n"
            f"- `max_depth`: Controls tree complexity ({MIN_TREE_DEPTH}-{MAX_TREE_DEPTH}). "
            f"Start with the default ({MAX_TREE_DEPTH}) and reduce if the rules are too detailed.\n"
            f"- `min_samples_leaf`: Minimum samples required at each leaf. Higher values "
            f"reduce noise. Default is {AUTO_MIN_SAMPLES_FLOOR}. Note: the tool automatically "
            "raises this value on large datasets (to roughly 2% of row count) to prevent "
            "overfitting — so the effective minimum may be higher than the value you supply.\n"
            "- `task`: Set to 'classification' or 'regression' to override auto-detection.\n\n"
            "**Interpreting results**:\n"
            "- `rules`: Each rule describes a path through the tree as a list of "
            "`Predicate` conditions (e.g., `tenure_months > 6`). The `prediction` "
            "field shows the predicted class or value at that leaf, and `confidence` "
            "(classification) or `std` (regression) indicates reliability.\n"
            "- `feature_importances`: Ranks features by how much they contributed to "
            "splits. Use the top features to guide further analysis or narrower trees.\n"
            "- `metrics`: For classification, `accuracy` (fraction of correct predictions); "
            "for regression, `r_squared` (variance explained by the tree, 0-1, higher is better) "
            "and `rmse` (average prediction error in the target's units).\n\n"
            "**Tips**:\n"
            "- Start with default depth and all features, then re-run with a smaller "
            "`max_depth` or a focused `features` list for simpler, clearer rules.\n"
            "- High-importance features are strong candidates for further segmentation "
            "or follow-up SQL analysis."
        )

    def get_tools(self) -> list[BaseTool]:
        """Return the `analyze_with_decision_tree` tool.

        Returns:
            list[BaseTool]: List containing the analyze_with_decision_tree tool.
        """
        return list(self._tools)

    def analyze_with_decision_tree(
        self,
        *,
        dataframe: str,
        target: str,
        features: list[str] | None = None,
        max_depth: int = MAX_TREE_DEPTH,
        min_samples_leaf: int = AUTO_MIN_SAMPLES_FLOOR,
        task: str | None = None,
    ) -> DecisionTreeResult | ToolCallError:
        """Analyze a DataFrame with a decision tree to discover column relationships.

        Args:
            dataframe (str): Name or ID of a registered DataFrame.
            target (str): Name of the target column to predict.
            features (list[str] | None): List of feature column names. If None,
                all suitable columns are used automatically.
            max_depth (int): Maximum tree depth (1-6). Default 6.
            min_samples_leaf (int): Minimum samples per leaf node. Default 5.
            task (str | None): Task type override: "classification", "regression",
                or None for auto-detect.

        Returns:
            DecisionTreeResult | ToolCallError: DecisionTreeResult with rules,
                feature importance, and metrics, or ToolCallError if the input
                is invalid.
        """
        logger.log(
            TOOL_CALL_LEVEL,
            "Tool call: analyze_with_decision_tree for dataframe={}, target={}",
            dataframe,
            target,
        )
        df = self._context.get_dataframe(dataframe)
        if isinstance(df, ToolCallError):
            logger.warning("analyze_with_decision_tree tool error: {} for dataframe={}", df.error_type, dataframe)
            return df

        try:
            result = _analyze_with_decision_tree(
                df,
                features,
                target,
                task=task,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
            )
        except ValueError as exc:
            # ValueErrors represent anticipated exceptions the LLM can adapt to
            # other exceptions are unanticipated and will be propagated
            error = ToolCallError(error_type="InvalidArgument", message=str(exc))
            logger.warning(
                "analyze_with_decision_tree tool error: {} for dataframe={}, target={}",
                error.error_type,
                dataframe,
                target,
            )
            return error

        logger.info(
            "Decision tree fitted: dataframe={}, target={}, task={}, depth={}, leaf_count={}, sample_count={}",
            dataframe,
            result.target,
            result.task,
            result.depth,
            result.leaf_count,
            result.sample_count,
        )
        return result

# endregion
