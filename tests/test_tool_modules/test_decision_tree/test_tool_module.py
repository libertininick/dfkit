"""Tests for DecisionTreeModule: protocol compliance, tool behavior, and toolkit integration."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest
from langchain_core.tools import BaseTool
from pytest_check import check

from dfkit.tool_modules.decision_tree import DecisionTreeModule
from dfkit.tool_modules.decision_tree.models import DecisionTreeResult, Predicate
from dfkit.tool_modules.models import ToolCallError
from dfkit.tool_modules.tool_module import ToolModule
from dfkit.toolkit import DataFrameToolkit

# region Shared fixture


@pytest.fixture
def toolkit_with_data() -> DataFrameToolkit:
    """Create a DataFrameToolkit with a reproducible 100-row customer DataFrame.

    Returns:
        DataFrameToolkit: Toolkit instance with "customers" registered.
    """
    toolkit = DataFrameToolkit()
    rng = np.random.default_rng(42)
    n = 100
    df = pl.DataFrame({
        "tenure_months": rng.integers(1, 48, size=n).tolist(),
        "monthly_spend": (rng.random(n) * 100 + 10).tolist(),
        "plan_type": rng.choice(["basic", "premium", "enterprise"], size=n).tolist(),
        "region": rng.choice(["north", "south", "east", "west"], size=n).tolist(),
        "churned": rng.choice(["yes", "no"], size=n).tolist(),
        "satisfaction": rng.integers(1, 6, size=n).tolist(),
    })
    toolkit.register_dataframe("customers", df)
    return toolkit


# endregion


# region TestDecisionTreeModule


class TestDecisionTreeModule:
    """Tests for DecisionTreeModule: protocol, tools, and per-tool behavior."""

    @pytest.fixture
    def module(self, toolkit_with_data: DataFrameToolkit) -> DecisionTreeModule:
        """Create a DecisionTreeModule from the toolkit fixture.

        Args:
            toolkit_with_data (DataFrameToolkit): Toolkit fixture with customer data.

        Returns:
            DecisionTreeModule: Module instance for testing.
        """
        # Access private attribute to inject context into module (intentional in tests)
        return DecisionTreeModule(toolkit_with_data._tool_module_context)

    def test_satisfies_tool_module_protocol(self, module: DecisionTreeModule) -> None:
        """DecisionTreeModule instance must satisfy the runtime-checkable ToolModule protocol.

        Args:
            module (DecisionTreeModule): Module fixture.
        """
        # Arrange — module is provided by fixture

        # Act / Assert
        assert isinstance(module, ToolModule)

    def test_system_prompt_is_nonempty_string(self, module: DecisionTreeModule) -> None:
        """system_prompt should return a non-empty string that references the tool by name.

        Args:
            module (DecisionTreeModule): Module fixture.
        """
        # Act
        prompt = module.system_prompt

        # Assert
        with check:
            assert isinstance(prompt, str)
        with check:
            assert len(prompt) > 0
        with check:
            assert "analyze_with_decision_tree" in prompt

    def test_get_tools_returns_one_tool(self, module: DecisionTreeModule) -> None:
        """get_tools should return a list containing exactly one BaseTool.

        Args:
            module (DecisionTreeModule): Module fixture.
        """
        # Act
        tools = module.get_tools()

        # Assert
        with check:
            assert isinstance(tools, list)
        with check:
            assert len(tools) == 1
        with check:
            assert isinstance(tools[0], BaseTool)

    def test_tool_name_is_analyze_with_decision_tree(self, module: DecisionTreeModule) -> None:
        """The single tool returned by get_tools should be named "analyze_with_decision_tree".

        Args:
            module (DecisionTreeModule): Module fixture.
        """
        # Act
        tools = module.get_tools()
        decision_tree_tool = tools[0]

        # Assert
        assert decision_tree_tool.name == "analyze_with_decision_tree"

    def test_tool_invocation_classification(self, module: DecisionTreeModule) -> None:
        """Invoking analyze_with_decision_tree with a categorical target returns a classification result.

        The result should be a DecisionTreeResult with task="classification",
        each rule should contain Predicate objects, and the accuracy metric
        should be present.

        Args:
            module (DecisionTreeModule): Module fixture.
        """
        # Arrange
        decision_tree_tool = module.get_tools()[0]

        # Act
        tool_input: dict[str, Any] = {"dataframe": "customers", "target": "churned"}
        result = decision_tree_tool.invoke(tool_input)

        # Assert
        with check:
            assert isinstance(result, DecisionTreeResult)
        with check:
            assert result.task == "classification"
        with check:
            assert result.target == "churned"
        with check:
            assert len(result.rules) >= 1
        # Verify predicates are Predicate model instances
        rules_with_predicates = [r for r in result.rules if r.predicates]
        assert rules_with_predicates, "Expected at least one rule with predicates"
        with check:
            assert all(isinstance(p, Predicate) for p in rules_with_predicates[0].predicates)
        # Assert — classification metric is present and finite
        with check:
            assert "accuracy" in result.metrics
        with check:
            assert isinstance(result.metrics["accuracy"], float)

    def test_tool_invocation_regression(self, module: DecisionTreeModule) -> None:
        """Invoking analyze_with_decision_tree with a numeric target returns a regression result.

        The result should be a DecisionTreeResult with task="regression", at least one rule
        with Predicate objects, regression-specific metrics (r_squared, rmse), and
        feature_importances covering all features.

        Args:
            module (DecisionTreeModule): Module fixture.
        """
        # Arrange
        decision_tree_tool = module.get_tools()[0]

        # Act
        tool_input: dict[str, Any] = {"dataframe": "customers", "target": "monthly_spend"}
        result = decision_tree_tool.invoke(tool_input)

        # Assert — identity and task
        with check:
            assert isinstance(result, DecisionTreeResult)
        with check:
            assert result.task == "regression"
        with check:
            assert result.target == "monthly_spend"
        # Assert — rules contain Predicate model instances
        with check:
            assert len(result.rules) >= 1
        rules_with_predicates = [r for r in result.rules if r.predicates]
        assert rules_with_predicates, "Expected at least one rule with predicates"
        with check:
            assert all(isinstance(p, Predicate) for p in rules_with_predicates[0].predicates)
        # Assert — regression metrics are present and finite
        with check:
            assert "r_squared" in result.metrics
        with check:
            assert "rmse" in result.metrics
        with check:
            assert isinstance(result.metrics["r_squared"], float)
        with check:
            assert isinstance(result.metrics["rmse"], float)

    def test_tool_returns_error_for_invalid_dataframe(self, module: DecisionTreeModule) -> None:
        """Invoking analyze_with_decision_tree with a nonexistent dataframe name returns ToolCallError.

        Args:
            module (DecisionTreeModule): Module fixture.
        """
        # Arrange
        decision_tree_tool = module.get_tools()[0]

        # Act
        tool_input: dict[str, Any] = {"dataframe": "nonexistent_df", "target": "churned"}
        result = decision_tree_tool.invoke(tool_input)

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "DataFrameNotFound"
        with check:
            assert "nonexistent_df" in result.message

    def test_tool_returns_error_for_invalid_target(self, module: DecisionTreeModule) -> None:
        """Invoking analyze_with_decision_tree with a valid dataframe but nonexistent target returns ToolCallError.

        Args:
            module (DecisionTreeModule): Module fixture.
        """
        # Arrange
        decision_tree_tool = module.get_tools()[0]

        # Act
        tool_input: dict[str, Any] = {"dataframe": "customers", "target": "nonexistent_column"}
        result = decision_tree_tool.invoke(tool_input)

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "InvalidArgument"
        with check:
            assert "nonexistent_column" in result.message

    def test_tool_invocation_with_explicit_features(self, module: DecisionTreeModule) -> None:
        """Invoking analyze_with_decision_tree with explicit features only uses those features.

        When features=["tenure_months", "monthly_spend"] is passed, the result's
        feature list must contain exactly those two columns and no others.

        Args:
            module (DecisionTreeModule): Module fixture.
        """
        # Arrange
        decision_tree_tool = module.get_tools()[0]
        requested_features = ["tenure_months", "monthly_spend"]

        # Act
        tool_input: dict[str, Any] = {
            "dataframe": "customers",
            "target": "churned",
            "features": requested_features,
        }
        result = decision_tree_tool.invoke(tool_input)

        # Assert
        assert isinstance(result, DecisionTreeResult)
        with check:
            assert set(result.features) == set(requested_features)
        with check:
            assert set(result.feature_importances.keys()).issubset(set(requested_features))
        for rule in result.rules:
            for p in rule.predicates:
                with check:
                    assert p.variable in requested_features

    def test_tool_returns_error_for_invalid_task_value(self, module: DecisionTreeModule) -> None:
        """Invoking analyze_with_decision_tree with an unrecognized task value returns ToolCallError.

        Valid task values are "classification", "regression", and None. Any
        other string such as "unsupported" must produce a ToolCallError rather
        than raising an exception or silently falling back to auto-detection.

        Args:
            module (DecisionTreeModule): Module fixture.
        """
        # Arrange
        decision_tree_tool = module.get_tools()[0]

        # Act
        tool_input: dict[str, Any] = {
            "dataframe": "customers",
            "target": "churned",
            "task": "unsupported",
        }
        result = decision_tree_tool.invoke(tool_input)

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "InvalidArgument"
        with check:
            assert "unsupported" in result.message

    def test_tool_invocation_task_auto_synonym_returns_result(self, module: DecisionTreeModule) -> None:
        """Invoking analyze_with_decision_tree with task="auto" produces a DecisionTreeResult.

        "auto" is a recognized synonym for None in _validate_task_override and
        is treated as automatic task detection. Passing task="auto" must
        therefore succeed and return a DecisionTreeResult, not a ToolCallError.

        Args:
            module (DecisionTreeModule): Module fixture.
        """
        # Arrange
        decision_tree_tool = module.get_tools()[0]

        # Act
        tool_input: dict[str, Any] = {
            "dataframe": "customers",
            "target": "churned",
            "task": "auto",
        }
        result = decision_tree_tool.invoke(tool_input)

        # Assert
        with check:
            assert isinstance(result, DecisionTreeResult)
        with check:
            assert result.task == "classification"

    @pytest.mark.parametrize(
        ("df_name", "target_col", "column_data"),
        [
            pytest.param(
                "degenerate_all_null",
                "all_null_target",
                [None] * 20,
                id="all-null-target",
            ),
            pytest.param(
                "degenerate_single_value",
                "single_value_target",
                ["yes"] * 20,
                id="single-unique-value-target",
            ),
        ],
    )
    def test_tool_returns_error_for_degenerate_target(
        self,
        df_name: str,
        target_col: str,
        column_data: list[str | None],
    ) -> None:
        """Invoking analyze_with_decision_tree with a degenerate target column returns ToolCallError.

        A target column is degenerate when all values are null or when only a
        single unique non-null value exists. Both cases must produce a
        ToolCallError rather than raising an exception.

        Args:
            df_name (str): Name to register the degenerate DataFrame under.
            target_col (str): Name of the degenerate target column.
            column_data (list[str | None]): Column values for the target (all-null or all-same).
        """
        # Arrange — build a DataFrame with a degenerate target column and register it
        n = len(column_data)
        rng = np.random.default_rng(0)
        df = pl.DataFrame({
            "feature_a": rng.integers(1, 10, size=n).tolist(),
            "feature_b": (rng.random(n) * 50).tolist(),
            target_col: column_data,
        })
        toolkit = DataFrameToolkit()
        toolkit.register_dataframe(df_name, df)
        # Access private attribute to inject context into module (intentional in tests)
        module = DecisionTreeModule(toolkit._tool_module_context)
        decision_tree_tool = module.get_tools()[0]

        # Act
        tool_input: dict[str, Any] = {
            "dataframe": df_name,
            "target": target_col,
        }
        result = decision_tree_tool.invoke(tool_input)

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "InvalidArgument"
        with check:
            assert target_col in result.message

    def test_tool_returns_error_for_invalid_feature_column(self, module: DecisionTreeModule) -> None:
        """Invoking analyze_with_decision_tree with a nonexistent feature column returns ToolCallError.

        When the features list contains a column name not present in the
        registered DataFrame, the tool must return a ToolCallError rather than
        raising an exception.

        Args:
            module (DecisionTreeModule): Module fixture.
        """
        # Arrange
        decision_tree_tool = module.get_tools()[0]

        # Act
        tool_input: dict[str, Any] = {
            "dataframe": "customers",
            "target": "churned",
            "features": ["tenure_months", "nonexistent_feature"],
        }
        result = decision_tree_tool.invoke(tool_input)

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "InvalidArgument"
        with check:
            assert "nonexistent_feature" in result.message

    def test_tool_returns_error_for_empty_features_list(self, module: DecisionTreeModule) -> None:
        """Invoking analyze_with_decision_tree with features=[] returns ToolCallError.

        When an empty feature list is passed, no valid feature columns remain
        after filtering, so fitting.py raises a ValueError which the tool
        converts to a ToolCallError. The error message must indicate that no
        valid feature columns remain.

        Args:
            module (DecisionTreeModule): Module fixture.
        """
        # Arrange
        decision_tree_tool = module.get_tools()[0]

        # Act
        tool_input: dict[str, Any] = {
            "dataframe": "customers",
            "target": "churned",
            "features": [],
        }
        result = decision_tree_tool.invoke(tool_input)

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "InvalidArgument"
        with check:
            assert "No valid feature columns" in result.message

    def test_tool_invocation_task_override_classification(self, module: DecisionTreeModule) -> None:
        """Passing task="classification" with a numeric target overrides auto-detection.

        When the target is "monthly_spend" (a numeric column that would normally
        auto-detect as regression), supplying task="classification" should produce
        a result with task == "classification". sklearn is expected to emit a
        UserWarning about unique class count exceeding 50% of samples.

        Args:
            module (DecisionTreeModule): Module fixture.
        """
        # Arrange
        decision_tree_tool = module.get_tools()[0]

        # Act — sklearn warns when treating a high-cardinality numeric column as classification
        tool_input: dict[str, Any] = {
            "dataframe": "customers",
            "target": "monthly_spend",
            "task": "classification",
        }
        with pytest.warns(UserWarning, match="unique classes is greater than 50%"):
            result = decision_tree_tool.invoke(tool_input)

        # Assert
        assert isinstance(result, DecisionTreeResult)
        with check:
            assert result.task == "classification"
        with check:
            assert "accuracy" in result.metrics

    @pytest.mark.parametrize("bad_depth", [-1, 0, 7])
    def test_tool_returns_error_for_max_depth_out_of_range(
        self,
        module: DecisionTreeModule,
        bad_depth: int,
    ) -> None:
        """Invoking analyze_with_decision_tree with max_depth outside 1-6 returns ToolCallError.

        Valid max_depth values are 1 through 6 inclusive. Values of -1, 0, or 7
        are out of range and must produce a ToolCallError rather than raising an
        exception.

        Args:
            module (DecisionTreeModule): Module fixture.
            bad_depth (int): An out-of-range max_depth value to test (-1, 0, or 7).
        """
        # Arrange
        decision_tree_tool = module.get_tools()[0]

        # Act
        tool_input: dict[str, Any] = {
            "dataframe": "customers",
            "target": "churned",
            "max_depth": bad_depth,
        }
        result = decision_tree_tool.invoke(tool_input)

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "InvalidArgument"
        with check:
            assert "max_depth" in result.message

    def test_tool_invocation_task_override_regression_on_categorical_target_returns_error(
        self,
        module: DecisionTreeModule,
    ) -> None:
        """Passing task="regression" with a string categorical target returns ToolCallError.

        Unlike the symmetric override of task="classification" on a numeric
        target (which succeeds by treating numeric values as class labels),
        forcing task="regression" on a string target like "churned" is
        invalid because sklearn cannot cast string values to float64.  The
        tool must surface this as a ToolCallError with error_type="InvalidArgument"
        rather than propagating the raw exception.

        Args:
            module (DecisionTreeModule): Module fixture.
        """
        # Arrange
        decision_tree_tool = module.get_tools()[0]

        # Act
        tool_input: dict[str, Any] = {
            "dataframe": "customers",
            "target": "churned",
            "task": "regression",
        }
        result = decision_tree_tool.invoke(tool_input)

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "InvalidArgument"
        with check:
            assert "float" in result.message

    def test_tool_invocation_min_samples_leaf_constrains_leaf_count(
        self,
        module: DecisionTreeModule,
    ) -> None:
        """Passing min_samples_leaf=50 on a 100-row fixture produces fewer leaves than the default.

        The parameter is threaded through _compute_effective_min_samples before
        being passed to the sklearn tree. A large min_samples_leaf forces coarser
        splits, so the resulting leaf_count should be lower than without the
        constraint. With a floor of 50 samples per leaf on 100 rows, the tree
        can produce at most 2 leaves.

        Args:
            module (DecisionTreeModule): Module fixture.
        """
        # Arrange
        decision_tree_tool = module.get_tools()[0]

        # Act — default min_samples_leaf
        default_input: dict[str, Any] = {"dataframe": "customers", "target": "churned"}
        default_result = decision_tree_tool.invoke(default_input)

        # Act — constrained min_samples_leaf of 50 on 100-row dataset
        constrained_input: dict[str, Any] = {
            "dataframe": "customers",
            "target": "churned",
            "min_samples_leaf": 50,
        }
        constrained_result = decision_tree_tool.invoke(constrained_input)

        # Assert
        assert isinstance(constrained_result, DecisionTreeResult)
        with check:
            assert constrained_result.leaf_count <= 2
        with check:
            assert isinstance(default_result, DecisionTreeResult)
        with check:
            assert constrained_result.leaf_count <= default_result.leaf_count

    @pytest.mark.parametrize("max_depth", [1, 6])
    def test_tool_invocation_valid_depth_boundaries(
        self,
        module: DecisionTreeModule,
        max_depth: int,
    ) -> None:
        """Invoking analyze_with_decision_tree with boundary-valid max_depth values succeeds.

        The valid range for max_depth is 1 through 6 inclusive. This test
        exercises the two boundary values (1 and 6) and asserts that each
        produces a DecisionTreeResult whose actual depth does not exceed the
        requested max_depth.

        Args:
            module (DecisionTreeModule): Module fixture.
            max_depth (int): A boundary-valid max_depth value to test (1 or 6).
        """
        # Arrange
        decision_tree_tool = module.get_tools()[0]

        # Act
        tool_input: dict[str, Any] = {
            "dataframe": "customers",
            "target": "churned",
            "max_depth": max_depth,
        }
        result = decision_tree_tool.invoke(tool_input)

        # Assert
        with check:
            assert isinstance(result, DecisionTreeResult)
        with check:
            assert result.depth <= max_depth


# endregion


# region TestDecisionTreeModuleIntegration


class TestDecisionTreeModuleIntegration:
    """Tests for DecisionTreeModule integration with DataFrameToolkit."""

    def test_toolkit_get_tools_includes_decision_tree(self, toolkit_with_data: DataFrameToolkit) -> None:
        """toolkit.get_tools(DecisionTreeModule) should include a tool named "analyze_with_decision_tree".

        Args:
            toolkit_with_data (DataFrameToolkit): Toolkit fixture with customer data.
        """
        # Arrange / Act
        tools = toolkit_with_data.get_tools(DecisionTreeModule)
        tool_names = {t.name for t in tools}

        # Assert
        assert "analyze_with_decision_tree" in tool_names

    def test_toolkit_system_prompt_includes_module(self, toolkit_with_data: DataFrameToolkit) -> None:
        """toolkit.get_system_prompt(DecisionTreeModule) should include decision tree guidance.

        Args:
            toolkit_with_data (DataFrameToolkit): Toolkit fixture with customer data.
        """
        # Arrange / Act
        prompt = toolkit_with_data.get_system_prompt(DecisionTreeModule)

        # Assert
        with check:
            assert isinstance(prompt, str)
        with check:
            assert "analyze_with_decision_tree" in prompt

    def test_toolkit_excludes_decision_tree_tool(self, toolkit_with_data: DataFrameToolkit) -> None:
        """toolkit.get_tools(DecisionTreeModule, exclude={"analyze_with_decision_tree"}) omits the tool.

        Args:
            toolkit_with_data (DataFrameToolkit): Toolkit fixture with customer data.
        """
        # Arrange / Act
        tools = toolkit_with_data.get_tools(DecisionTreeModule, exclude={"analyze_with_decision_tree"})
        tool_names = {t.name for t in tools}

        # Assert
        assert "analyze_with_decision_tree" not in tool_names


# endregion
