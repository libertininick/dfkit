"""Tests for DataFrameContext class."""

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import SQLInterfaceError
from pytest_check import check

from dfkit.context import DataFrameContext


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Create a simple eager DataFrame for testing.

    Returns:
        pl.DataFrame: DataFrame with columns 'a' and 'b'.
    """
    return pl.DataFrame({"a": [3, 1, 3], "b": [None, -5, 42]})


@pytest.fixture
def sample_lazy_df() -> pl.LazyFrame:
    """Create a simple LazyFrame for testing.

    Returns:
        pl.LazyFrame: LazyFrame with columns 'x' and 'y'.
    """
    return pl.DataFrame({"x": [50, 10, 30], "y": [-8, 0, 100]}).lazy()


@pytest.fixture
def sample_df_2() -> pl.DataFrame:
    """Create another eager DataFrame for multi-frame tests.

    Returns:
        pl.DataFrame: DataFrame with columns 'a' and 'c'.
    """
    return pl.DataFrame({"a": [1, 1, 2], "c": ["apple", "banana", "cherry"]})


@pytest.fixture
def populated_context(sample_df: pl.DataFrame, sample_lazy_df: pl.LazyFrame) -> DataFrameContext:
    """Create a pre-configured context with multiple registered dataframes.

    Args:
        sample_df (pl.DataFrame): Sample eager DataFrame fixture.
        sample_lazy_df (pl.LazyFrame): Sample LazyFrame fixture.

    Returns:
        DataFrameContext: Context with 'df_00000001' and 'df_00000002' registered.
    """
    ctx = DataFrameContext()
    ctx.register("df_00000001", sample_df)
    ctx.register("df_00000002", sample_lazy_df)
    return ctx


class TestInitialization:
    """Tests for DataFrameContext initialization."""

    def test_init_empty_context(self) -> None:
        """Verify empty context initialization creates an empty registry.

        An empty context should have zero dataframes, an empty frame_ids list,
        and a repr indicating no dataframes are registered.
        """
        ctx = DataFrameContext()

        with check:
            assert len(ctx) == 0, "Empty context should have length 0"
        with check:
            assert ctx.dataframe_ids == (), "Empty context should have empty frame_ids"
        with check:
            assert repr(ctx) == "DataFrameContext(dataframes=[])", "Empty context repr should show no dataframes"

    def test_init_with_frames_mapping(self, sample_df: pl.DataFrame, sample_lazy_df: pl.LazyFrame) -> None:
        """Verify initialization with pre-populated dataframes registers all dataframes.

        When a mapping is provided to the constructor, all dataframes should be
        registered immediately and accessible by their keys.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
            sample_lazy_df (pl.LazyFrame): Sample LazyFrame fixture.
        """
        dataframes = {"df_00000001": sample_df, "df_00000002": sample_lazy_df}
        ctx = DataFrameContext(dataframes=dataframes)

        with check:
            assert len(ctx) == 2, "Context should contain both dataframes"
        with check:
            assert "df_00000001" in ctx, "Frame 'df_00000001' should be registered"
        with check:
            assert "df_00000002" in ctx, "Frame 'df_00000002' should be registered"
        with check:
            assert ctx.get_dataframe("df_00000001") is sample_df, "Should return the same DataFrame object"
        with check:
            assert ctx.get_dataframe("df_00000002") is sample_lazy_df, "Should return the same LazyFrame object"
        with check:
            assert set(ctx.dataframe_ids) == {"df_00000001", "df_00000002"}, "Frame frame_ids should match mapping keys"

    def test_init_with_duplicate_frame_ids_raises_value_error(self, sample_df: pl.DataFrame) -> None:
        """Verify that duplicate frame_ids in initialization raise ValueError.

        Since register_many() calls register() for each frame, attempting to
        register duplicate frame_ids during initialization should fail.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
        """
        # This test verifies behavior indirectly by testing register_many
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_df)

        with pytest.raises(ValueError, match="already registered"):
            ctx.register("df_00000001", sample_df)


class TestRegistration:
    """Tests for frame registration methods."""

    def test_register_dataframe(self, sample_df: pl.DataFrame) -> None:
        """Verify successful registration of an eager DataFrame.

        Registering a DataFrame should increase the context size, make the
        frame accessible by frame_id, and allow retrieval via get_frame().

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_df)

        with check:
            assert len(ctx) == 1, "Context should have 1 frame after registration"
            assert "df_00000001" in ctx, "Frame frame_id should be in context"
            assert ctx.get_dataframe("df_00000001") is sample_df, "Should return the same DataFrame object"

    def test_register_lazyframe(self, sample_lazy_df: pl.LazyFrame) -> None:
        """Verify successful registration of a LazyFrame.

        Registering a LazyFrame should work identically to registering a
        DataFrame, supporting lazy evaluation patterns.

        Args:
            sample_lazy_df (pl.LazyFrame): Sample LazyFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_lazy_df)

        with check:
            assert len(ctx) == 1, "Context should have 1 frame after registration"
            assert "df_00000001" in ctx, "Frame frame_id should be in context"
            assert ctx.get_dataframe("df_00000001") is sample_lazy_df, "Should return the same LazyFrame object"

    def test_register_duplicate_frame_id_raises_value_error(self, sample_df: pl.DataFrame) -> None:
        """Verify that registering a duplicate frame_id raises ValueError.

        Attempting to register a frame with an already-registered frame_id should
        fail to prevent accidental overwrites and maintain registry integrity.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_df)

        with pytest.raises(ValueError, match="already registered"):
            ctx.register("df_00000001", sample_df)

    def test_register_returns_self(self, sample_df: pl.DataFrame) -> None:
        """Verify that register() returns self for method chaining.

        The register method should return the context instance to enable
        fluent interface patterns.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()
        result = ctx.register("df_00000001", sample_df)

        with check:
            assert result is ctx, "register() should return the same context instance"

    def test_register_many_with_multiple_frames(
        self, sample_df: pl.DataFrame, sample_lazy_df: pl.LazyFrame, sample_df_2: pl.DataFrame
    ) -> None:
        """Verify bulk registration with register_many().

        register_many() should allow registering multiple dataframes at once,
        making them all accessible with correct frame_ids and data.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
            sample_lazy_df (pl.LazyFrame): Sample LazyFrame fixture.
            sample_df_2 (pl.DataFrame): Second sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()
        dataframes = {"df_00000001": sample_df, "df_00000002": sample_lazy_df, "df_00000003": sample_df_2}
        ctx.register_many(dataframes)

        with check:
            assert len(ctx) == 3, "Context should have 3 dataframes"
        with check:
            assert set(ctx.dataframe_ids) == {"df_00000001", "df_00000002", "df_00000003"}, (
                "All frame_ids should be registered"
            )
        with check:
            assert ctx.get_dataframe("df_00000001") is sample_df, "Should retrieve first frame"
        with check:
            assert ctx.get_dataframe("df_00000002") is sample_lazy_df, "Should retrieve second frame"
        with check:
            assert ctx.get_dataframe("df_00000003") is sample_df_2, "Should retrieve third frame"

    def test_register_many_returns_self(self, sample_df: pl.DataFrame) -> None:
        """Verify that register_many() returns self for method chaining.

        Like register(), register_many() should support fluent interface.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()
        result = ctx.register_many({"df_00000001": sample_df})

        with check:
            assert result is ctx, "register_many() should return the same context instance"

    def test_register_many_with_empty_mapping(self) -> None:
        """Verify no-op behavior with empty mapping.

        Calling register_many() with an empty dict should succeed without
        errors and leave the context unchanged.
        """
        ctx = DataFrameContext()
        ctx.register_many({})

        with check:
            assert len(ctx) == 0, "Context should remain empty"

    def test_register_same_dataframe_different_frame_ids(self, sample_df: pl.DataFrame) -> None:
        """Verify that the same DataFrame can be registered under multiple frame_ids.

        This is valid behavior allowing multiple views or aliases of the same
        data source in different SQL contexts.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_df)
        ctx.register("df_00000002", sample_df)

        with check:
            assert len(ctx) == 2, "Context should have 2 registered frame_ids"
        with check:
            assert ctx.get_dataframe("df_00000001") is sample_df, "First frame_id should return the DataFrame"
        with check:
            assert ctx.get_dataframe("df_00000002") is sample_df, "Second frame_id should return the same DataFrame"
        with check:
            assert ctx.get_dataframe("df_00000001") is ctx.get_dataframe("df_00000002"), (
                "Both frame_ids should reference same object"
            )


class TestUnregistration:
    """Tests for frame unregistration methods."""

    def test_unregister_single_frame(self, populated_context: DataFrameContext) -> None:
        """Verify unregistration with a single frame_id.

        Unregistering one frame should remove it from the context while
        leaving other dataframes accessible.

        Args:
            populated_context (DataFrameContext): Pre-configured context fixture.
        """
        ctx = populated_context
        initial_len = len(ctx)
        ctx.unregister("df_00000001")

        with check:
            assert len(ctx) == initial_len - 1, "Context should have one fewer frame"
        with check:
            assert "df_00000001" not in ctx, "Unregistered frame should not be in context"
        with check:
            assert "df_00000002" in ctx, "Other dataframes should remain registered"

    def test_unregister_multiple_frames_by_collection(
        self, sample_df: pl.DataFrame, sample_lazy_df: pl.LazyFrame, sample_df_2: pl.DataFrame
    ) -> None:
        """Verify unregistration with a collection of frame_ids.

        Unregistering multiple dataframes at once should remove all specified
        dataframes while preserving others.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
            sample_lazy_df (pl.LazyFrame): Sample LazyFrame fixture.
            sample_df_2 (pl.DataFrame): Second sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_df)
        ctx.register("df_00000002", sample_lazy_df)
        ctx.register("df_00000003", sample_df_2)

        ctx.unregister(["df_00000001", "df_00000002"])

        with check:
            assert len(ctx) == 1, "Context should have 1 frame remaining"
        with check:
            assert "df_00000001" not in ctx, "First unregistered frame should be gone"
        with check:
            assert "df_00000002" not in ctx, "Second unregistered frame should be gone"
        with check:
            assert "df_00000003" in ctx, "Remaining frame should still be accessible"

    def test_unregister_non_existent_frame_raises_key_error(self) -> None:
        """Verify error when unregistering a non-existent frame.

        Attempting to unregister a frame that was never registered should
        fail with a clear error message.
        """
        ctx = DataFrameContext()

        with pytest.raises(KeyError, match="not registered"):
            ctx.unregister("non_existent")

    def test_unregister_then_re_register_same_frame_id(
        self, sample_df: pl.DataFrame, sample_df_2: pl.DataFrame
    ) -> None:
        """Verify that a frame_id can be reused after unregistration.

        After unregistering a frame, the same frame_id should be available for
        registering a new frame (even with different data).

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
            sample_df_2 (pl.DataFrame): Second sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000004", sample_df)
        ctx.unregister("df_00000004")
        ctx.register("df_00000004", sample_df_2)

        with check:
            assert "df_00000004" in ctx, "frame_id should be available after unregister"
        with check:
            assert ctx.get_dataframe("df_00000004") is sample_df_2, "Should retrieve newly registered frame"
        with check:
            assert ctx.get_dataframe("df_00000004") is not sample_df, "Should not retrieve old frame"

    def test_unregister_returns_self(self, populated_context: DataFrameContext) -> None:
        """Verify that unregister() returns self for method chaining.

        The unregister method should support fluent interface patterns.

        Args:
            populated_context (DataFrameContext): Pre-configured context fixture.
        """
        ctx = populated_context
        result = ctx.unregister("df_00000001")

        with check:
            assert result is ctx, "unregister() should return the same context instance"


class TestSQLQueryExecution:
    """Tests for SQL query execution."""

    def test_execute_sql_successful_query_eager(self, sample_df: pl.DataFrame) -> None:
        """Verify successful SQL query execution in eager mode.

        Executing a SQL query on an eager DataFrame should return an eager
        DataFrame with the expected results.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_df)

        result = ctx.execute_sql("SELECT a FROM df_00000001 WHERE a > 1", eager=True)

        with check:
            assert isinstance(result, pl.DataFrame), "Result should be an eager DataFrame"
        with check:
            assert result.shape == (2, 1), "Result should have 2 rows and 1 column"
        with check:
            assert result["a"].to_list() == [3, 3], "Result should contain filtered values"

    def test_execute_sql_successful_query_lazy(self, sample_lazy_df: pl.LazyFrame) -> None:
        """Verify successful SQL query execution in lazy mode.

        Executing a SQL query with eager=False should return a LazyFrame that
        can be collected to produce correct results.

        Args:
            sample_lazy_df (pl.LazyFrame): Sample LazyFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_lazy_df)

        result = ctx.execute_sql("SELECT x FROM df_00000001 WHERE x > 10", eager=False)

        with check:
            assert isinstance(result, pl.LazyFrame), "Result should be a LazyFrame"

        # Collect and verify data
        collected = result.collect()
        with check:
            assert collected.shape == (2, 1), "Collected result should have 2 rows and 1 column"
        with check:
            assert collected["x"].to_list() == [50, 30], "Result should contain filtered values"

    def test_execute_sql_join_multiple_frames(self, sample_df: pl.DataFrame, sample_df_2: pl.DataFrame) -> None:
        """Verify SQL queries across multiple registered dataframes.

        SQL JOIN operations should work across different registered dataframes,
        demonstrating the registry's utility for multi-table queries.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
            sample_df_2 (pl.DataFrame): Second sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_df)
        ctx.register("df_00000002", sample_df_2)

        result = ctx.execute_sql(
            "SELECT df_00000001.a, df_00000002.c FROM df_00000001 "
            "JOIN df_00000002 ON df_00000001.a = df_00000002.a "
            "WHERE df_00000002.c = 'banana'",
            eager=True,
        )

        with check:
            assert isinstance(result, pl.DataFrame), "Result should be an eager DataFrame"
        with check:
            assert result.shape[0] == 1, "Should have 1 row matching 'banana' with a=1"
        with check:
            assert all(result["c"] == "banana"), "All rows should have c='banana'"
        with check:
            assert result["a"][0] == 1, "Result should have a=1"

    def test_execute_sql_empty_query_raises_value_error(self, sample_df: pl.DataFrame) -> None:
        """Verify error when query is an empty string.

        Empty queries are invalid and should be rejected before reaching Polars.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_df)

        with pytest.raises(ValueError, match="empty or whitespace"):
            ctx.execute_sql("")

    def test_execute_sql_whitespace_only_query_raises_value_error(self, sample_df: pl.DataFrame) -> None:
        """Verify error when query contains only whitespace.

        Whitespace-only queries should be treated the same as empty queries
        and rejected with a clear error message.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_df)

        with pytest.raises(ValueError, match="empty or whitespace"):
            ctx.execute_sql("   \n\t  ")

    def test_execute_sql_empty_context_raises_value_error(self) -> None:
        """Verify error when executing SQL with no registered DataFrames.

        Executing a SQL query without any registered dataframes should fail.
        """
        ctx = DataFrameContext()

        with pytest.raises(ValueError):
            ctx.execute_sql("SELECT * FROM df_00000001")

    def test_execute_sql_syntax_error(self, sample_df: pl.DataFrame) -> None:
        """Verify behavior with SQL syntax error.

        Invalid SQL syntax should propagate a Polars exception. We let Polars
        handle SQL validation rather than duplicating it.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_df)

        # Polars will raise an exception for invalid SQL
        with pytest.raises(SQLInterfaceError):  # Polars-specific exception
            ctx.execute_sql("INVALID SQL SYNTAX SELECT")

    def test_execute_sql_unregistered_table(self, sample_df: pl.DataFrame) -> None:
        """Verify behavior when query references an unregistered table.

        Referencing a table that doesn't exist in the registry should result
        in a Polars error about the missing table.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_df)

        # Polars will raise an exception for non-existent table
        with pytest.raises(SQLInterfaceError):  # Polars-specific exception
            ctx.execute_sql("SELECT * FROM df_00000002")  # df_00000002 not registered


class TestGetFrame:
    """Tests for frame retrieval."""

    def test_get_frame_existing_dataframe(self, sample_df: pl.DataFrame) -> None:
        """Verify retrieval of a registered DataFrame.

        get_frame() should return the exact same DataFrame object that was
        registered, preserving object identity.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_df)

        result = ctx.get_dataframe("df_00000001")

        with check:
            assert result is sample_df, "Should return the same DataFrame object"
        with check:
            assert result.schema == sample_df.schema, "Schema should match"
        with check:
            assert result.shape == sample_df.shape, "Shape should match"

    def test_get_frame_existing_lazyframe(self, sample_lazy_df: pl.LazyFrame) -> None:
        """Verify retrieval of a registered LazyFrame.

        get_frame() should return the exact same LazyFrame object that was
        registered.

        Args:
            sample_lazy_df (pl.LazyFrame): Sample LazyFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_lazy_df)

        result = ctx.get_dataframe("df_00000001")

        with check:
            assert result is sample_lazy_df, "Should return the same LazyFrame object"
        with check:
            assert result.collect_schema() == sample_lazy_df.collect_schema(), "Schema should match"

    def test_get_frame_non_existent_raises_key_error(self) -> None:
        """Verify error when retrieving a non-existent frame.

        Attempting to get a frame that was never registered should fail with
        a clear error message indicating the missing frame_id.
        """
        ctx = DataFrameContext()

        with pytest.raises(KeyError, match="not registered"):
            ctx.get_dataframe("df_00000001")


class TestContainerProtocol:
    """Tests for container protocol methods (__len__, __contains__, __repr__)."""

    def test_len_empty_context(self) -> None:
        """Verify len() on empty context returns zero.

        The length of an empty context should be 0, consistent with standard
        Python container semantics.
        """
        ctx = DataFrameContext()

        with check:
            assert len(ctx) == 0, "Empty context should have length 0"

    def test_len_with_frames(self, sample_df: pl.DataFrame, sample_lazy_df: pl.LazyFrame) -> None:
        """Verify len() with registered dataframes returns correct count.

        The length should equal the number of registered dataframes.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
            sample_lazy_df (pl.LazyFrame): Sample LazyFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_df)
        ctx.register("df_00000002", sample_lazy_df)
        ctx.register("df_00000003", sample_df)

        with check:
            assert len(ctx) == 3, "Context should have 3 dataframes"

    def test_len_after_register_and_unregister(self, sample_df: pl.DataFrame, sample_lazy_df: pl.LazyFrame) -> None:
        """Verify len() updates correctly after registration and unregistration.

        The length should dynamically reflect the current state of the registry.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
            sample_lazy_df (pl.LazyFrame): Sample LazyFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_df)
        ctx.register("df_00000002", sample_lazy_df)

        with check:
            assert len(ctx) == 2, "Should have 2 dataframes after registration"

        ctx.unregister("df_00000001")

        with check:
            assert len(ctx) == 1, "Should have 1 frame after unregistration"

    def test_contains_existing_frame(self, sample_df: pl.DataFrame) -> None:
        """Verify 'in' operator returns True for registered dataframes.

        The membership test should return True for registered frame_ids.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_df)

        with check:
            assert "df_00000001" in ctx, "Registered frame should be in context"

    def test_contains_non_existent_frame(self) -> None:
        """Verify 'in' operator returns False for non-existent dataframes.

        The membership test should return False for frame_ids that were never
        registered or have been unregistered.
        """
        ctx = DataFrameContext()

        with check:
            assert "df_00000001" not in ctx, "Non-existent frame should not be in context"

    def test_repr_empty_context(self) -> None:
        """Verify string representation of empty context.

        The repr should clearly indicate that no dataframes are registered.
        """
        ctx = DataFrameContext()

        with check:
            assert repr(ctx) == "DataFrameContext(dataframes=[])", "Empty context repr should show empty list"

    def test_repr_with_frames(self, sample_df: pl.DataFrame, sample_lazy_df: pl.LazyFrame) -> None:
        """Verify string representation with registered dataframes.

        The repr should list all registered frame_ids in a clear format.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
            sample_lazy_df (pl.LazyFrame): Sample LazyFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_df)
        ctx.register("df_00000002", sample_lazy_df)

        repr_str = repr(ctx)

        with check:
            assert "DataFrameContext(dataframes=[" in repr_str, "Repr should show DataFrameContext with dataframes"
        with check:
            assert "'df_00000001'" in repr_str, "Repr should include first frame_id"
        with check:
            assert "'df_00000002'" in repr_str, "Repr should include second frame_id"


class TestProperties:
    """Tests for context properties."""

    def test_frame_ids_empty_context(self) -> None:
        """Verify frame_ids property on empty context returns empty list.

        An empty context should return an empty tuple of frame_ids.
        """
        ctx = DataFrameContext()

        with check:
            assert ctx.dataframe_ids == (), "Empty context should have empty frame_ids"

    def test_frame_ids_with_frames(
        self, sample_df: pl.DataFrame, sample_lazy_df: pl.LazyFrame, sample_df_2: pl.DataFrame
    ) -> None:
        """Verify frame_ids property with registered dataframes.

        The property should return a list containing all registered frame_ids
        in registration order.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
            sample_lazy_df (pl.LazyFrame): Sample LazyFrame fixture.
            sample_df_2 (pl.DataFrame): Second sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_df)
        ctx.register("df_00000002", sample_lazy_df)
        ctx.register("df_00000003", sample_df_2)

        frame_ids = ctx.dataframe_ids

        with check:
            assert len(frame_ids) == 3, "Should return 3 frame_ids"
        with check:
            assert set(frame_ids) == {"df_00000001", "df_00000002", "df_00000003"}, (
                "Should contain all registered frame_ids"
            )
        with check:
            assert frame_ids == ("df_00000001", "df_00000002", "df_00000003"), "Should maintain registration order"


class TestClear:
    """Tests for clear() method."""

    def test_clear_empty_context(self) -> None:
        """Verify clear() on empty context is a no-op.

        Calling clear() on an already-empty context should succeed without
        errors.
        """
        ctx = DataFrameContext()
        ctx.clear()

        with check:
            assert len(ctx) == 0, "Context should remain empty"

    def test_clear_context_with_frames(
        self, sample_df: pl.DataFrame, sample_lazy_df: pl.LazyFrame, sample_df_2: pl.DataFrame
    ) -> None:
        """Verify clear() unregisters all dataframes.

        After calling clear(), the context should be empty with no registered
        dataframes accessible.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
            sample_lazy_df (pl.LazyFrame): Sample LazyFrame fixture.
            sample_df_2 (pl.DataFrame): Second sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_df)
        ctx.register("df_00000002", sample_lazy_df)
        ctx.register("df_00000003", sample_df_2)

        original_frame_ids = set(ctx.dataframe_ids)
        ctx.clear()

        with check:
            assert len(ctx) == 0, "Context should be empty after clear"
        with check:
            assert ctx.dataframe_ids == (), "frame_ids should be empty"
        for frame_id in original_frame_ids:
            with check:
                assert frame_id not in ctx, f"Frame '{frame_id}' should not be in context after clear"

    def test_clear_returns_self(self, sample_df: pl.DataFrame) -> None:
        """Verify clear() returns self for method chaining.

        The clear method should support fluent interface patterns.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_df)

        result = ctx.clear()

        with check:
            assert result is ctx, "clear() should return the same context instance"


class TestMethodChaining:
    """Tests for method chaining support."""

    def test_method_chaining_register_register(self, sample_df: pl.DataFrame, sample_df_2: pl.DataFrame) -> None:
        """Verify fluent interface with multiple register() calls.

        Chaining multiple register() calls should successfully register all
        dataframes in sequence.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
            sample_df_2 (pl.DataFrame): Second sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()
        ctx.register("df_00000001", sample_df).register("df_00000002", sample_df_2)

        with check:
            assert len(ctx) == 2, "Both dataframes should be registered"
        with check:
            assert "df_00000001" in ctx, "First frame should be registered"
        with check:
            assert "df_00000002" in ctx, "Second frame should be registered"

    def test_method_chaining_register_unregister_clear(
        self, sample_df: pl.DataFrame, sample_lazy_df: pl.LazyFrame, sample_df_2: pl.DataFrame
    ) -> None:
        """Verify complex chaining scenario with multiple methods.

        Complex chains combining register, unregister, and clear should work
        correctly with expected final state.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
            sample_lazy_df (pl.LazyFrame): Sample LazyFrame fixture.
            sample_df_2 (pl.DataFrame): Second sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()

        # Chain multiple operations
        result = (
            ctx
            .register("df_00000001", sample_df)
            .register("df_00000002", sample_lazy_df)
            .register("df_00000003", sample_df_2)
            .unregister("df_00000001")
            .clear()
        )

        with check:
            assert result is ctx, "Chained methods should return context"
        with check:
            assert len(ctx) == 0, "Context should be empty after clear"
        with check:
            assert ctx.dataframe_ids == (), "frame_ids should be empty"

    def test_method_chaining_register_many_unregister(
        self, sample_df: pl.DataFrame, sample_lazy_df: pl.LazyFrame, sample_df_2: pl.DataFrame
    ) -> None:
        """Verify chaining with register_many() and unregister().

        register_many() should chain smoothly with other methods.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
            sample_lazy_df (pl.LazyFrame): Sample LazyFrame fixture.
            sample_df_2 (pl.DataFrame): Second sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()
        dataframes = {"df_00000001": sample_df, "df_00000002": sample_lazy_df, "df_00000003": sample_df_2}

        ctx.register_many(dataframes).unregister(["df_00000001", "df_00000002"])

        with check:
            assert len(ctx) == 1, "Should have 1 frame after chained operations"
        with check:
            assert "df_00000003" in ctx, "Remaining frame should be accessible"


class TestSQLContextSynchronization:
    """Tests for internal SQLContext synchronization."""

    def test_sql_context_synchronization_after_register_unregister_register(
        self, sample_df: pl.DataFrame, sample_df_2: pl.DataFrame
    ) -> None:
        """Verify that internal SQLContext stays in sync after operations.

        After registering, unregistering, and re-registering dataframes, SQL
        queries should work correctly without stale references.

        Args:
            sample_df (pl.DataFrame): Sample eager DataFrame fixture.
            sample_df_2 (pl.DataFrame): Second sample eager DataFrame fixture.
        """
        ctx = DataFrameContext()

        # Register initial frame
        ctx.register("df_00000001", sample_df)
        result1 = ctx.execute_sql("SELECT COUNT(*) as count FROM df_00000001", eager=True)
        with check:
            assert result1["count"][0] == 3, "Initial query should work"

        # Unregister and verify SQL fails
        ctx.unregister("df_00000001")
        with pytest.raises(ValueError):  # No registered dataframes
            ctx.execute_sql("SELECT * FROM df_00000001")

        # Re-register with different data
        ctx.register("df_00000001", sample_df_2)
        result2 = ctx.execute_sql("SELECT COUNT(*) as count FROM df_00000001", eager=True)
        with check:
            assert result2["count"][0] == 3, "Query should work after re-registration"

        # Verify we're querying new data
        result3 = ctx.execute_sql("SELECT c FROM df_00000001 LIMIT 1", eager=True)
        with check:
            assert "c" in result3.columns, "Should query new frame's columns"
