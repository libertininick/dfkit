"""Tests for ToolModuleContext class."""

from __future__ import annotations

from typing import NamedTuple

import polars as pl
import pytest
from pytest_check import check

from dfkit.models import DataFrameReference, ToolCallError
from dfkit.tool_module_context import ToolModuleContext
from dfkit.toolkit import DataFrameToolkit


class ContextFixture(NamedTuple):
    """Fixture data for ToolModuleContext tests.

    Attributes:
        context (ToolModuleContext): The tool module context instance.
        toolkit (DataFrameToolkit): The toolkit that created the context.
        ref (DataFrameReference): The reference for the "sales" DataFrame.
    """

    context: ToolModuleContext
    toolkit: DataFrameToolkit
    ref: DataFrameReference


@pytest.fixture
def context_with_data() -> ContextFixture:
    """Create a ToolModuleContext with registered DataFrames using a DataFrameToolkit.

    This fixture demonstrates the intended usage pattern: the toolkit creates
    the context and passes its own methods for reference resolution and
    name validation. Use this fixture when testing ToolModuleContext methods
    directly. For toolkit-level composition tests, create a toolkit inline.

    Returns:
        ContextFixture: Named tuple with context, toolkit, and sales reference.
    """
    # Arrange - create toolkit and register a DataFrame
    toolkit = DataFrameToolkit()
    sales_df = pl.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})
    ref = toolkit.register_dataframe("sales", sales_df)

    # Use the toolkit's own context instance (created during __init__)
    context = toolkit._tool_module_context

    return ContextFixture(context=context, toolkit=toolkit, ref=ref)


class TestReferencesProperty:
    """Tests for ToolModuleContext.references property."""

    def test_references_returns_empty_tuple_when_empty(self) -> None:
        """Verify references returns empty tuple when no DataFrames registered.

        When the registry is empty (no DataFrames registered), the references
        property should return an empty tuple.
        """
        # Arrange
        toolkit = DataFrameToolkit()
        context = toolkit._tool_module_context

        # Act
        references = context.references

        # Assert
        with check:
            assert isinstance(references, tuple), "references should return a tuple"
        with check:
            assert len(references) == 0, "should be empty when registry is empty"

    def test_references_returns_all_registered(self, context_with_data: ContextFixture) -> None:
        """Verify references returns tuple of all registered references.

        The references property should return a tuple containing all
        DataFrameReference objects currently registered in the toolkit.

        Args:
            context_with_data (ContextFixture): Fixture providing context, toolkit, and reference.
        """
        # Arrange
        context = context_with_data.context
        ref = context_with_data.ref

        # Act
        references = context.references

        # Assert
        with check:
            assert isinstance(references, tuple), "references should return a tuple"
        with check:
            assert len(references) == 1, "should have exactly one reference"
        with check:
            assert references[0] is ref, "should contain the registered reference"


class TestGetDataFrame:
    """Tests for ToolModuleContext.get_dataframe method."""

    def test_get_dataframe_by_name(self, context_with_data: ContextFixture) -> None:
        """Verify get_dataframe retrieves DataFrame by name.

        When given a registered DataFrame name, get_dataframe should return
        the actual pl.DataFrame object.

        Args:
            context_with_data (ContextFixture): Fixture providing context, toolkit, and reference.
        """
        # Arrange
        context = context_with_data.context

        # Act
        result = context.get_dataframe("sales")

        # Assert
        with check:
            assert isinstance(result, pl.DataFrame), "should return a DataFrame"
        with check:
            assert result.columns == ["id", "amount"], "should have correct columns"
        with check:
            assert len(result) == 3, "should have 3 rows"

    def test_get_dataframe_by_id(self, context_with_data: ContextFixture) -> None:
        """Verify get_dataframe retrieves DataFrame by ID.

        When given a registered DataFrame ID, get_dataframe should return
        the actual pl.DataFrame object.

        Args:
            context_with_data (ContextFixture): Fixture providing context, toolkit, and reference.
        """
        # Arrange
        context = context_with_data.context
        ref = context_with_data.ref

        # Act
        result = context.get_dataframe(ref.id)

        # Assert
        with check:
            assert isinstance(result, pl.DataFrame), "should return a DataFrame"
        with check:
            assert result.columns == ["id", "amount"], "should have correct columns"

    def test_get_dataframe_not_found(self, context_with_data: ContextFixture) -> None:
        """Verify get_dataframe returns ToolCallError for unknown identifier.

        When given an identifier that doesn't match any registered DataFrame,
        get_dataframe should return a ToolCallError with error_type "DataFrameNotFound".

        Args:
            context_with_data (ContextFixture): Fixture providing context, toolkit, and reference.
        """
        # Arrange
        context = context_with_data.context

        # Act
        result = context.get_dataframe("nonexistent")

        # Assert
        with check:
            assert isinstance(result, ToolCallError), "should return ToolCallError"
        with check:
            assert result.error_type == "DataFrameNotFound", "should have correct error type"
        with check:
            assert "nonexistent" in result.message, "error message should mention the identifier"


class TestGetDataframeReference:
    """Tests for ToolModuleContext.get_dataframe_reference method."""

    def test_get_dataframe_reference_by_name(self, context_with_data: ContextFixture) -> None:
        """Verify get_dataframe_reference returns DataFrameReference by name.

        When given a registered DataFrame name, get_dataframe_reference should return
        the DataFrameReference with that name.

        Args:
            context_with_data (ContextFixture): Fixture providing context, toolkit, and reference.
        """
        # Arrange
        context = context_with_data.context
        ref = context_with_data.ref

        # Act
        result = context.get_dataframe_reference("sales")

        # Assert
        with check:
            assert isinstance(result, DataFrameReference), "should return DataFrameReference"
        with check:
            assert result.name == "sales", "should have correct name"
        with check:
            assert result.id == ref.id, "should have correct ID"

    def test_get_dataframe_reference_by_id(self, context_with_data: ContextFixture) -> None:
        """Verify get_dataframe_reference returns DataFrameReference by ID.

        When given a registered DataFrame ID, get_dataframe_reference should return
        the matching DataFrameReference.

        Args:
            context_with_data (ContextFixture): Fixture providing context, toolkit, and reference.
        """
        # Arrange
        context = context_with_data.context
        ref = context_with_data.ref

        # Act
        result = context.get_dataframe_reference(ref.id)

        # Assert
        with check:
            assert isinstance(result, DataFrameReference), "should return DataFrameReference"
        with check:
            assert result.id == ref.id, "should have matching ID"
        with check:
            assert result.name == "sales", "should have correct name"

    def test_get_dataframe_reference_not_found(self, context_with_data: ContextFixture) -> None:
        """Verify get_dataframe_reference returns ToolCallError for unknown identifier.

        When given an identifier that doesn't match any registered DataFrame,
        get_dataframe_reference should return a ToolCallError with error_type "DataFrameNotFound".

        Args:
            context_with_data (ContextFixture): Fixture providing context, toolkit, and reference.
        """
        # Arrange
        context = context_with_data.context

        # Act
        result = context.get_dataframe_reference("nonexistent")

        # Assert
        with check:
            assert isinstance(result, ToolCallError), "should return ToolCallError"
        with check:
            assert result.error_type == "DataFrameNotFound", "should have correct error type"


class TestRegisterDataFrame:
    """Tests for ToolModuleContext.register_dataframe method."""

    def test_register_valid_name(self, context_with_data: ContextFixture) -> None:
        """Verify register_dataframe creates a new DataFrame with a valid name.

        When given a valid name that doesn't conflict with existing names,
        register_dataframe should create and return a DataFrameReference.

        Args:
            context_with_data (ContextFixture): Fixture providing context, toolkit, and reference.
        """
        # Arrange
        context = context_with_data.context
        new_df = pl.DataFrame({"product": ["A", "B", "C"], "price": [10.0, 20.0, 30.0]})

        # Act
        result = context.register_dataframe("new_result", new_df)

        # Assert
        with check:
            assert isinstance(result, DataFrameReference), "should return DataFrameReference"
        with check:
            assert result.name == "new_result", "should have correct name"
        with check:
            assert isinstance(result.id, str), "should have an ID"
        with check:
            assert result.id.startswith("df_"), "ID should match pattern df_xxxxxxxx"

    def test_register_duplicate_name(self, context_with_data: ContextFixture) -> None:
        """Verify register returns ToolCallError for duplicate name.

        When given a name that is already registered, register should return
        a ToolCallError with error_type "DuplicateName".

        Args:
            context_with_data (ContextFixture): Fixture providing context, toolkit, and reference.
        """
        # Arrange
        context = context_with_data.context
        duplicate_df = pl.DataFrame({"status": ["pending", "approved"]})

        # Act
        result = context.register_dataframe("sales", duplicate_df)

        # Assert
        with check:
            assert isinstance(result, ToolCallError), "should return ToolCallError"
        with check:
            assert result.error_type == "DuplicateName", "should have correct error type"
        with check:
            assert "sales" in result.message, "error message should mention the duplicate name"

    def test_register_id_pattern_name(self, context_with_data: ContextFixture) -> None:
        """Verify register rejects names matching the ID pattern.

        When given a name that looks like a DataFrame ID (df_xxxxxxxx), register
        should return a ToolCallError with error_type "InvalidArgument".

        Args:
            context_with_data (ContextFixture): Fixture providing context, toolkit, and reference.
        """
        # Arrange
        context = context_with_data.context
        new_df = pl.DataFrame({"value": [1]})

        # Act
        result = context.register_dataframe("df_1a2b3c4d", new_df)

        # Assert
        with check:
            assert isinstance(result, ToolCallError), "should return ToolCallError"
        with check:
            assert result.error_type == "InvalidArgument", "should have correct error type"
        with check:
            assert "df_1a2b3c4d" in result.message, "error message should mention the invalid name"

    def test_register_with_lineage(self, context_with_data: ContextFixture) -> None:
        """Verify register sets lineage metadata correctly.

        When given parent_ids and source_query, register should create a
        DataFrameReference with those lineage fields set.

        Args:
            context_with_data (ContextFixture): Fixture providing context, toolkit, and reference.
        """
        # Arrange
        context = context_with_data.context
        ref = context_with_data.ref
        derived_df = pl.DataFrame({"id": [1, 2], "filtered_amount": [100, 200]})
        # Safe string interpolation - ref.id is a validated DataFrameId, not user input
        source_query = f"SELECT * FROM {ref.id} WHERE amount > 50"  # noqa: S608 - ref.id is a validated DataFrameId, not user input

        # Act
        result = context.register_dataframe(
            "high_sales",
            derived_df,
            parent_ids=[ref.id],
            source_query=source_query,
        )

        # Assert
        with check:
            assert isinstance(result, DataFrameReference), "should return DataFrameReference"
        with check:
            assert result.parent_ids == [ref.id], "should have correct parent_ids"
        with check:
            assert result.source_query == source_query, "should have correct source_query"

    def test_register_with_description(self, context_with_data: ContextFixture) -> None:
        """Verify register sets description metadata correctly.

        When given description parameter, register should create a
        DataFrameReference with that description field set.

        Args:
            context_with_data (ContextFixture): Fixture providing context, toolkit, and reference.
        """
        # Arrange
        context = context_with_data.context
        new_df = pl.DataFrame({"product": ["Widget", "Gadget"], "price": [9.99, 19.99]})
        description = "Product catalog with pricing information"

        # Act
        result = context.register_dataframe(
            "products",
            new_df,
            description=description,
        )

        # Assert
        with check:
            assert isinstance(result, DataFrameReference), "should return DataFrameReference"
        with check:
            assert result.description == description, "should have correct description"

    def test_register_with_column_descriptions(self, context_with_data: ContextFixture) -> None:
        """Verify register sets column_descriptions metadata correctly.

        When given column_descriptions parameter, register should create a
        DataFrameReference with those column descriptions set in the column summaries.

        Args:
            context_with_data (ContextFixture): Fixture providing context, toolkit, and reference.
        """
        # Arrange
        context = context_with_data.context
        new_df = pl.DataFrame({"user_id": [101, 102], "score": [85.5, 92.3]})
        column_descriptions = {
            "user_id": "Unique identifier for the user",
            "score": "Performance score as a percentage",
        }

        # Act
        result = context.register_dataframe(
            "user_scores",
            new_df,
            column_descriptions=column_descriptions,
        )

        # Assert
        with check:
            assert isinstance(result, DataFrameReference), "should return DataFrameReference"
        with check:
            assert result.column_summaries["user_id"].description == column_descriptions["user_id"], (
                "should have correct user_id description"
            )
        with check:
            assert result.column_summaries["score"].description == column_descriptions["score"], (
                "should have correct score description"
            )

    def test_register_with_all_optional_parameters(self, context_with_data: ContextFixture) -> None:
        """Verify register handles all optional parameters together.

        When given description, parent_ids, source_query, and column_descriptions,
        register should create a DataFrameReference with all metadata fields set.

        Args:
            context_with_data (ContextFixture): Fixture providing context, toolkit, and reference.
        """
        # Arrange
        context = context_with_data.context
        ref = context_with_data.ref
        derived_df = pl.DataFrame({"product": ["Widget"], "total_sales": [500]})
        description = "Aggregated sales by product"
        column_descriptions = {
            "product": "Product name",
            "total_sales": "Total revenue for this product",
        }
        source_query = f"SELECT product, SUM(amount) AS total_sales FROM {ref.id} GROUP BY product"  # noqa: S608 - ref.id is a validated DataFrameId, not user input

        # Act
        result = context.register_dataframe(
            "product_totals",
            derived_df,
            description=description,
            parent_ids=[ref.id],
            source_query=source_query,
            column_descriptions=column_descriptions,
        )

        # Assert
        with check:
            assert isinstance(result, DataFrameReference), "should return DataFrameReference"
        with check:
            assert result.description == description, "should have correct description"
        with check:
            assert result.parent_ids == [ref.id], "should have correct parent_ids"
        with check:
            assert result.source_query == source_query, "should have correct source_query"
        with check:
            assert result.column_summaries["product"].description == column_descriptions["product"], (
                "should have correct product description"
            )
        with check:
            assert result.column_summaries["total_sales"].description == column_descriptions["total_sales"], (
                "should have correct total_sales description"
            )


class TestRegisteredDataFrameVisibility:
    """Tests for visibility of registered DataFrames."""

    def test_registered_dataframe_visible_in_references(self, context_with_data: ContextFixture) -> None:
        """Verify registered DataFrame appears in references property.

        After registering a new DataFrame via context.register(), it should
        appear in the tuple returned by context.references.

        Args:
            context_with_data (ContextFixture): Fixture providing context, toolkit, and reference.
        """
        # Arrange
        context = context_with_data.context
        new_df = pl.DataFrame({"measurement": [10.5, 20.3, 30.1], "unit": ["kg", "kg", "kg"]})

        # Act
        new_ref = context.register_dataframe("new_data", new_df)
        references = context.references

        # Assert
        with check:
            assert len(references) == 2, "should have two references after registration"
        with check:
            assert new_ref in references, "new reference should be in references tuple"

    def test_registered_dataframe_accessible_via_get_dataframe(self, context_with_data: ContextFixture) -> None:
        """Verify registered DataFrame is accessible via get_dataframe.

        After registering a new DataFrame, it should be retrievable by both
        name and ID using get_dataframe().

        Args:
            context_with_data (ContextFixture): Fixture providing context, toolkit, and reference.
        """
        # Arrange
        context = context_with_data.context
        new_df = pl.DataFrame({"sensor": ["temp", "humidity"], "reading": [22.5, 65.0]})

        # Act
        new_ref = context.register_dataframe("new_data", new_df)
        if isinstance(new_ref, ToolCallError):
            pytest.fail(f"register_dataframe failed with error: {new_ref.message}")
        else:
            result_by_name = context.get_dataframe("new_data")
            result_by_id = context.get_dataframe(new_ref.id)

        # Assert
        with check:
            assert isinstance(result_by_name, pl.DataFrame), "should retrieve by name"
        with check:
            assert isinstance(result_by_id, pl.DataFrame), "should retrieve by ID"
        with check:
            assert result_by_name.columns == ["sensor", "reading"], "retrieved DataFrame should have correct columns"
        with check:
            assert len(result_by_name) == 2, "retrieved DataFrame should have correct row count"


class TestNoUnregisterOrClearMethods:
    """Tests verifying ToolModuleContext does not expose destructive operations."""

    def test_no_unregister_or_clear_methods(self, context_with_data: ContextFixture) -> None:
        """Verify ToolModuleContext does not have unregister or clear methods.

        The ToolModuleContext should be additive-only, providing no way for
        modules to remove or clear DataFrames from the registry.

        Args:
            context_with_data (ContextFixture): Fixture providing context, toolkit, and reference.
        """
        # Arrange
        context = context_with_data.context

        # Assert
        with check:
            assert not hasattr(context, "unregister"), "should not have unregister method"
        with check:
            assert not hasattr(context, "clear"), "should not have clear method"
