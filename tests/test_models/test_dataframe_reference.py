"""Tests for the DataFrameReference model."""

from __future__ import annotations

import json
import re

import polars as pl
import pytest
from pydantic import ValidationError
from pytest_check import check

from dfkit.models import DataFrameReference


class TestDataFrameReference:
    """Tests for the DataFrameReference model."""

    # -------------------------------------------------------------------------
    # from_dataframe factory method tests
    # -------------------------------------------------------------------------

    def test_from_dataframe_minimal_arguments_creates_valid_reference(self) -> None:
        """Given DataFrame and name only, When from_dataframe called, Then creates reference with defaults."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        # Act
        ref = DataFrameReference.from_dataframe("test_df", df)

        # Assert
        with check:
            assert ref.name == "test_df"
        with check:
            assert not ref.description
        with check:
            assert ref.num_rows == 3
        with check:
            assert ref.num_columns == 2
        with check:
            assert ref.column_names == ["a", "b"]
        with check:
            assert ref.parent_ids == []
        with check:
            assert ref.source_query is None

    def test_from_dataframe_all_arguments_creates_reference_with_all_values(self) -> None:
        """Given all arguments, When from_dataframe called, Then creates reference with all values."""
        # Arrange
        df = pl.DataFrame({"col1": [10, 20], "col2": [1.5, 2.5]})
        parent_ids = ["df_00000001", "df_00000002"]

        # Act
        ref = DataFrameReference.from_dataframe(
            "derived_df",
            df,
            description="A derived DataFrame",
            column_descriptions={"col1": "Integer column", "col2": "Float column"},
            parent_ids=parent_ids,
            source_query="SELECT col1, col2 FROM base",
        )

        # Assert
        with check:
            assert ref.name == "derived_df"
        with check:
            assert ref.description == "A derived DataFrame"
        with check:
            assert ref.parent_ids == parent_ids
        with check:
            assert ref.source_query == "SELECT col1, col2 FROM base"
        with check:
            assert ref.column_summaries["col1"].description == "Integer column"
        with check:
            assert ref.column_summaries["col2"].description == "Float column"

    def test_from_dataframe_generates_unique_id(self) -> None:
        """Given same DataFrame, When from_dataframe called twice, Then generates different ids."""
        # Arrange
        df = pl.DataFrame({"a": [1]})

        # Act
        ref1 = DataFrameReference.from_dataframe("test", df)
        ref2 = DataFrameReference.from_dataframe("test", df)

        # Assert
        with check:
            assert ref1.id != ref2.id
        with check:
            assert ref1.id.startswith("df_")
        with check:
            assert len(ref1.id) == 11  # "df_" + 8 hex chars

    def test_from_dataframe_empty_dataframe_creates_valid_reference(self) -> None:
        """Given empty DataFrame, When from_dataframe called, Then creates reference with None min/max."""
        # Arrange
        df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64), "b": pl.Series([], dtype=pl.Utf8)})

        # Act
        ref = DataFrameReference.from_dataframe("empty_df", df)

        # Assert
        with check:
            assert ref.num_rows == 0
        with check:
            assert ref.num_columns == 2
        with check:
            assert ref.column_summaries["a"].min is None
        with check:
            assert ref.column_summaries["a"].max is None
        with check:
            assert ref.column_summaries["b"].min is None
        with check:
            assert ref.column_summaries["b"].max is None

    def test_from_dataframe_with_null_values_creates_valid_reference(self) -> None:
        """Given DataFrame with nulls, When from_dataframe called, Then creates reference with null counts."""
        # Arrange
        df = pl.DataFrame({"a": [1, None, 3], "b": [None, None, "z"]})

        # Act
        ref = DataFrameReference.from_dataframe("null_df", df)

        # Assert
        with check:
            assert ref.num_rows == 3
        with check:
            assert ref.column_summaries["a"].null_count == 1
        with check:
            assert ref.column_summaries["b"].null_count == 2

    def test_from_dataframe_single_column_single_row(self) -> None:
        """Given DataFrame with single column and row, When from_dataframe called, Then creates valid reference."""
        # Arrange
        df = pl.DataFrame({"only_col": [42]})

        # Act
        ref = DataFrameReference.from_dataframe("single", df)

        # Assert
        with check:
            assert ref.num_rows == 1
        with check:
            assert ref.num_columns == 1
        with check:
            assert ref.column_names == ["only_col"]

    def test_from_dataframe_partial_column_descriptions(self) -> None:
        """Given column descriptions for some columns only, When from_dataframe called, Then others empty."""
        # Arrange
        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})

        # Act
        ref = DataFrameReference.from_dataframe(
            "partial_desc",
            df,
            column_descriptions={"a": "Column A description"},
        )

        # Assert
        with check:
            assert ref.column_summaries["a"].description == "Column A description"
        with check:
            assert not ref.column_summaries["b"].description
        with check:
            assert not ref.column_summaries["c"].description

    # -------------------------------------------------------------------------
    # Field tests
    # -------------------------------------------------------------------------

    def test_id_field_follows_dataframe_id_pattern(self) -> None:
        """Given DataFrameReference, When id accessed, Then follows df_<8 hex chars> pattern."""
        # Arrange
        df = pl.DataFrame({"a": [1]})

        # Act
        ref = DataFrameReference.from_dataframe("test", df)

        # Assert
        pattern = re.compile(r"^df_[0-9a-f]{8}$")
        with check:
            assert pattern.match(ref.id) is not None

    def test_name_field_preserves_value(self) -> None:
        """Given name with special characters, When from_dataframe called, Then name preserved exactly."""
        # Arrange
        df = pl.DataFrame({"a": [1]})
        name = "my-test_df.2024"

        # Act
        ref = DataFrameReference.from_dataframe(name, df)

        # Assert
        with check:
            assert ref.name == name

    def test_description_field_empty_string_when_none(self) -> None:
        """Given no description, When from_dataframe called, Then description is empty string."""
        # Arrange
        df = pl.DataFrame({"a": [1]})

        # Act
        ref = DataFrameReference.from_dataframe("test", df)

        # Assert
        with check:
            assert not ref.description

    def test_description_field_preserves_value(self) -> None:
        """Given description, When from_dataframe called, Then description preserved."""
        # Arrange
        df = pl.DataFrame({"a": [1]})
        desc = "This DataFrame contains sales data for Q4 2024."

        # Act
        ref = DataFrameReference.from_dataframe("test", df, description=desc)

        # Assert
        with check:
            assert ref.description == desc

    def test_num_rows_matches_dataframe_height(self) -> None:
        """Given DataFrame with specific height, When from_dataframe called, Then num_rows matches."""
        # Arrange
        df = pl.DataFrame({"a": list(range(100))})

        # Act
        ref = DataFrameReference.from_dataframe("test", df)

        # Assert
        with check:
            assert ref.num_rows == 100

    def test_num_columns_matches_dataframe_width(self) -> None:
        """Given DataFrame with specific width, When from_dataframe called, Then num_columns matches."""
        # Arrange
        df = pl.DataFrame({f"col_{i}": [1] for i in range(5)})

        # Act
        ref = DataFrameReference.from_dataframe("test", df)

        # Assert
        with check:
            assert ref.num_columns == 5

    def test_column_names_matches_dataframe_columns(self) -> None:
        """Given DataFrame with columns, When from_dataframe called, Then column_names matches order."""
        # Arrange
        df = pl.DataFrame({"z": [1], "a": [2], "m": [3]})

        # Act
        ref = DataFrameReference.from_dataframe("test", df)

        # Assert
        with check:
            assert ref.column_names == ["z", "a", "m"]

    def test_column_summaries_contains_all_columns(self) -> None:
        """Given DataFrame with multiple columns, When from_dataframe called, Then column_summaries has all."""
        # Arrange
        df = pl.DataFrame({"int_col": [1, 2, 3], "str_col": ["a", "b", "c"], "float_col": [1.1, 2.2, 3.3]})

        # Act
        ref = DataFrameReference.from_dataframe("test", df)

        # Assert
        with check:
            assert set(ref.column_summaries.keys()) == {"int_col", "str_col", "float_col"}

    def test_parent_ids_default_empty_list(self) -> None:
        """Given no parent_ids, When from_dataframe called, Then parent_ids is empty list."""
        # Arrange
        df = pl.DataFrame({"a": [1]})

        # Act
        ref = DataFrameReference.from_dataframe("test", df)

        # Assert
        with check:
            assert ref.parent_ids == []

    def test_parent_ids_preserves_values(self) -> None:
        """Given parent_ids and source_query, When from_dataframe called, Then parent_ids preserved."""
        # Arrange
        df = pl.DataFrame({"a": [1]})
        parent_ids = ["df_11111111", "df_22222222", "df_33333333"]

        # Act
        ref = DataFrameReference.from_dataframe("test", df, parent_ids=parent_ids, source_query="SELECT * FROM base")

        # Assert
        with check:
            assert ref.parent_ids == parent_ids

    # -------------------------------------------------------------------------
    # source_query field tests (existing tests)
    # -------------------------------------------------------------------------

    def test_source_query_default_none(self) -> None:
        """Given DataFrameReference without source_query, When checked, Then source_query is None."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})

        # Act
        ref = DataFrameReference.from_dataframe("test", df)

        # Assert
        with check:
            assert ref.source_query is None

    def test_source_query_with_value(self) -> None:
        """Given source_query and parent_ids, When checked, Then source_query contains SQL string."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})
        sql = "SELECT * FROM base_table WHERE value > 0"

        # Act
        ref = DataFrameReference.from_dataframe("derived", df, source_query=sql, parent_ids=["df_00000001"])

        # Assert
        with check:
            assert ref.source_query == sql

    def test_source_query_serialization(self) -> None:
        """Given DataFrameReference with source_query, When serialized, Then source_query is included."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})
        sql = "SELECT a FROM parent"

        # Act
        ref = DataFrameReference.from_dataframe("derived", df, source_query=sql, parent_ids=["df_00000001"])
        ref_dict = ref.model_dump()

        # Assert
        with check:
            assert "source_query" in ref_dict
        with check:
            assert ref_dict["source_query"] == sql

    def test_source_query_json_serialization(self) -> None:
        """Given DataFrameReference with source_query, When serialized to JSON, Then source_query is included."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})
        sql = "SELECT * FROM base"

        # Act
        ref = DataFrameReference.from_dataframe("derived", df, source_query=sql, parent_ids=["df_00000001"])
        json_str = ref.model_dump_json()

        # Assert
        with check:
            assert "source_query" in json_str
        with check:
            assert sql in json_str

    def test_source_query_none_serialization(self) -> None:
        """Given DataFrameReference without source_query, When serialized, Then source_query is None in dict."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})

        # Act
        ref = DataFrameReference.from_dataframe("base", df)
        ref_dict = ref.model_dump()

        # Assert
        with check:
            assert "source_query" in ref_dict
        with check:
            assert ref_dict["source_query"] is None

    # -------------------------------------------------------------------------
    # Base/derivative consistency validation tests
    # -------------------------------------------------------------------------

    def test_source_query_without_parent_ids_raises(self) -> None:
        """Given source_query but no parent_ids, When constructing, Then raises ValidationError."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})

        # Act/Assert
        with pytest.raises(ValidationError, match="has source_query but empty parent_ids"):
            DataFrameReference.from_dataframe("derived", df, source_query="SELECT * FROM base")

    def test_parent_ids_without_source_query_raises(self) -> None:
        """Given parent_ids but no source_query, When constructing, Then raises ValidationError."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})

        # Act/Assert
        with pytest.raises(ValidationError, match="has parent_ids but no source_query"):
            DataFrameReference.from_dataframe("derived", df, parent_ids=["df_00000001"])

    def test_base_reference_no_parent_ids_no_source_query_succeeds(self) -> None:
        """Given no parent_ids and no source_query, When constructing, Then succeeds as base reference."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})

        # Act
        ref = DataFrameReference.from_dataframe("base", df)

        # Assert
        with check:
            assert ref.parent_ids == []
        with check:
            assert ref.source_query is None

    def test_derivative_reference_with_both_succeeds(self) -> None:
        """Given parent_ids and source_query, When constructing, Then succeeds as derivative reference."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})

        # Act
        ref = DataFrameReference.from_dataframe(
            "derived", df, parent_ids=["df_00000001"], source_query="SELECT * FROM base"
        )

        # Assert
        with check:
            assert ref.parent_ids == ["df_00000001"]
        with check:
            assert ref.source_query == "SELECT * FROM base"

    # -------------------------------------------------------------------------
    # Serialization tests
    # -------------------------------------------------------------------------

    def test_model_dump_contains_all_fields(self) -> None:
        """Given DataFrameReference, When model_dump called, Then all fields present."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})
        ref = DataFrameReference.from_dataframe("test", df)

        # Act
        ref_dict = ref.model_dump()

        # Assert
        expected_fields = [
            "id",
            "name",
            "description",
            "num_rows",
            "num_columns",
            "column_names",
            "column_summaries",
            "parent_ids",
            "source_query",
        ]
        for field in expected_fields:
            with check:
                assert field in ref_dict, f"Field '{field}' missing from model_dump"

    def test_model_dump_json_produces_valid_json(self) -> None:
        """Given DataFrameReference, When model_dump_json called, Then produces valid JSON string."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        ref = DataFrameReference.from_dataframe("test", df, description="Test DataFrame")

        # Act
        json_str = ref.model_dump_json()

        # Assert
        with check:
            assert isinstance(json_str, str)
        # Should be parseable JSON
        parsed = json.loads(json_str)
        with check:
            assert parsed["name"] == "test"
        with check:
            assert parsed["description"] == "Test DataFrame"
        with check:
            assert parsed["num_rows"] == 3

    def test_model_dump_json_with_indent_produces_formatted_output(self) -> None:
        """Given DataFrameReference, When model_dump_json with indent, Then output is formatted."""
        # Arrange
        df = pl.DataFrame({"a": [1]})
        ref = DataFrameReference.from_dataframe("test", df)

        # Act
        compact = ref.model_dump_json()
        formatted = ref.model_dump_json(indent=2)

        # Assert
        with check:
            assert "\n" not in compact
        with check:
            assert "\n" in formatted

    def test_json_round_trip_preserves_all_fields(self) -> None:
        """Given DataFrameReference, When serialized and deserialized, Then all fields preserved."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]})
        original = DataFrameReference.from_dataframe(
            "test_df",
            df,
            description="A test DataFrame",
            parent_ids=["df_aaaaaaaa", "df_bbbbbbbb"],
            source_query="SELECT * FROM parent",
        )

        # Act
        json_str = original.model_dump_json()
        restored = DataFrameReference.model_validate_json(json_str)

        # Assert
        with check:
            assert restored.id == original.id
        with check:
            assert restored.name == original.name
        with check:
            assert restored.description == original.description
        with check:
            assert restored.num_rows == original.num_rows
        with check:
            assert restored.num_columns == original.num_columns
        with check:
            assert restored.column_names == original.column_names
        with check:
            assert restored.parent_ids == original.parent_ids
        with check:
            assert restored.source_query == original.source_query

    def test_model_dump_column_summaries_structure(self) -> None:
        """Given DataFrameReference, When model_dump called, Then column_summaries has correct structure."""
        # Arrange
        df = pl.DataFrame({"int_col": [1, 2, 3]})
        ref = DataFrameReference.from_dataframe("test", df, column_descriptions={"int_col": "Integer values"})

        # Act
        ref_dict = ref.model_dump()

        # Assert
        summary = ref_dict["column_summaries"]["int_col"]
        expected_summary_fields = [
            "description",
            "dtype",
            "count",
            "null_count",
            "unique_count",
            "min",
            "max",
            "mean",
            "std",
            "p25",
            "p50",
            "p75",
        ]
        for field in expected_summary_fields:
            with check:
                assert field in summary, f"ColumnSummary field '{field}' missing"
        with check:
            assert summary["description"] == "Integer values"

    # -------------------------------------------------------------------------
    # Schema and field descriptions tests
    # -------------------------------------------------------------------------

    def test_all_fields_have_descriptions_in_schema(self) -> None:
        """Given DataFrameReference model, When schema inspected, Then all fields have descriptions."""
        # Arrange/Act
        schema = DataFrameReference.model_json_schema()
        properties = schema["properties"]

        # Assert
        expected_fields = [
            "id",
            "name",
            "description",
            "num_rows",
            "num_columns",
            "column_names",
            "column_summaries",
            "parent_ids",
            "source_query",
        ]
        for field in expected_fields:
            with check:
                assert field in properties, f"Field '{field}' not in schema properties"
            with check:
                assert "description" in properties.get(field, {}), f"Field '{field}' does not have description"

    def test_source_query_has_description_in_schema(self) -> None:
        """Given DataFrameReference model, When schema inspected, Then source_query has description."""
        # Arrange/Act
        schema = DataFrameReference.model_json_schema()
        properties = schema["properties"]

        # Assert
        with check:
            assert "source_query" in properties
        with check:
            assert "description" in properties["source_query"]

    # -------------------------------------------------------------------------
    # Edge case tests
    # -------------------------------------------------------------------------

    def test_dataframe_with_special_column_names(self) -> None:
        """Given DataFrame with special column names, When from_dataframe called, Then names preserved."""
        # Arrange
        df = pl.DataFrame({"column with spaces": [1], "column-with-dashes": [2], "123_numeric_start": [3]})

        # Act
        ref = DataFrameReference.from_dataframe("special", df)

        # Assert
        with check:
            assert "column with spaces" in ref.column_names
        with check:
            assert "column-with-dashes" in ref.column_names
        with check:
            assert "123_numeric_start" in ref.column_names
        with check:
            assert "column with spaces" in ref.column_summaries
        with check:
            assert "column-with-dashes" in ref.column_summaries

    def test_dataframe_with_various_dtypes(self) -> None:
        """Given DataFrame with various dtypes, When from_dataframe called, Then column_summaries created."""
        # Arrange
        df = pl.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        })

        # Act
        ref = DataFrameReference.from_dataframe("multi_dtype", df)

        # Assert
        with check:
            assert len(ref.column_summaries) == 4
        with check:
            assert "Int64" in ref.column_summaries["int_col"].dtype
        with check:
            assert "Float64" in ref.column_summaries["float_col"].dtype
        with check:
            assert "String" in ref.column_summaries["str_col"].dtype
        with check:
            assert "Boolean" in ref.column_summaries["bool_col"].dtype

    def test_dataframe_with_all_null_column_creates_valid_reference(self) -> None:
        """Given DataFrame with all-null column, When from_dataframe called, Then min/max are None."""
        # Arrange
        df = pl.DataFrame({"all_null": [None, None, None], "has_values": [1, 2, 3]})

        # Act
        ref = DataFrameReference.from_dataframe("with_nulls", df)

        # Assert
        with check:
            assert ref.num_rows == 3
        with check:
            assert ref.column_summaries["all_null"].min is None
        with check:
            assert ref.column_summaries["all_null"].max is None
        with check:
            assert ref.column_summaries["all_null"].null_count == 3
        with check:
            assert ref.column_summaries["has_values"].min == 1
        with check:
            assert ref.column_summaries["has_values"].max == 3

    def test_dataframe_with_partial_null_column_creates_valid_reference(self) -> None:
        """Given DataFrame with some nulls in column, When from_dataframe called, Then creates reference."""
        # Arrange
        df = pl.DataFrame({"some_null": [1, None, 3], "no_null": [1, 2, 3]})

        # Act
        ref = DataFrameReference.from_dataframe("with_some_nulls", df)

        # Assert
        with check:
            assert ref.column_summaries["some_null"].null_count == 1
        with check:
            assert ref.column_summaries["some_null"].count == 2
        with check:
            assert ref.column_summaries["no_null"].null_count == 0
        with check:
            assert ref.column_summaries["no_null"].count == 3
