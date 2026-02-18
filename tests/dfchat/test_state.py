"""Unit tests for dfchat.state session state management functions.

Tests cover all public functions that read from and write to ``st.session_state``.
Each test uses ``streamlit.testing.v1.AppTest`` to run functions inside a real,
isolated Streamlit runtime — no mocking required.
"""

from __future__ import annotations

import pytest
from pytest_check import check
from streamlit.testing.v1 import AppTest

from dfkit import DataFrameToolkit

# ---------------------------------------------------------------------------
# init_session_state
# ---------------------------------------------------------------------------


class TestInitSessionState:
    """Tests for init_session_state()."""

    def test_init_session_state_creates_all_keys(self) -> None:
        """All six expected keys should be present with correct default types after init.

        Verifies that toolkit is a DataFrameToolkit, chat_history is an empty list,
        agent is None, agent_stale is True, and both df name keys are None.
        """
        # Arrange
        at = AppTest.from_string("""
            from dfchat.state import init_session_state
            init_session_state()
        """)

        # Act
        at.run()

        # Assert
        with check:
            assert not at.exception, f"Unexpected Streamlit exception: {at.exception}"
        with check:
            assert isinstance(at.session_state["toolkit"], DataFrameToolkit), (
                "toolkit should be a DataFrameToolkit instance"
            )
        with check:
            assert at.session_state["chat_history"] == [], "chat_history should default to empty list"
        with check:
            assert at.session_state["agent"] is None, "agent should default to None"
        with check:
            assert at.session_state["agent_stale"] is True, "agent_stale should default to True"
        with check:
            assert at.session_state["selected_df"] is None, "selected_df should default to None"
        with check:
            assert at.session_state["recommended_df"] is None, "recommended_df should default to None"

    def test_init_session_state_idempotent_preserves_existing_toolkit(self) -> None:
        """Calling init_session_state twice should not replace an already-set toolkit.

        The function is documented as idempotent: a key that already exists in
        session state must not be overwritten.  This test runs the app script twice
        (the second run reuses the same AppTest session state) and asserts the
        toolkit object identity is preserved.
        """
        # Arrange — first run populates defaults
        at = AppTest.from_string("""
            from dfchat.state import init_session_state
            init_session_state()
        """)
        at.run()
        assert not at.exception, f"First run raised an unexpected exception: {at.exception}"
        original_toolkit = at.session_state["toolkit"]

        # Act — second run must be a no-op for existing keys
        at.run()

        # Assert
        with check:
            assert not at.exception, f"Second run raised an unexpected exception: {at.exception}"
        with check:
            assert at.session_state["toolkit"] is original_toolkit, (
                "Second init_session_state call should not replace existing toolkit"
            )

    def test_init_session_state_idempotent_preserves_custom_value(self) -> None:
        """Pre-seeded state values should survive a call to init_session_state.

        When a caller has already set a key (e.g. from a previous page load),
        init_session_state must leave it unchanged.
        """
        # Arrange — pre-populate selected_df before the app script runs
        at = AppTest.from_string("""
            from dfchat.state import init_session_state
            init_session_state()
        """)
        at.session_state["selected_df"] = "my_frame"

        # Act
        at.run()

        # Assert
        with check:
            assert not at.exception, f"Unexpected Streamlit exception: {at.exception}"
        with check:
            assert at.session_state["selected_df"] == "my_frame", (
                "Pre-existing selected_df should not be overwritten by init_session_state"
            )


# ---------------------------------------------------------------------------
# get_toolkit
# ---------------------------------------------------------------------------


class TestGetToolkit:
    """Tests for get_toolkit()."""

    def test_get_toolkit_returns_dataframe_toolkit_instance(self) -> None:
        """get_toolkit() should return the DataFrameToolkit stored during init."""
        # Arrange
        at = AppTest.from_string("""
            import streamlit as st
            from dfchat.state import init_session_state, get_toolkit
            init_session_state()
            st.session_state["_toolkit_type"] = type(get_toolkit()).__name__
            st.session_state["_same_object"] = get_toolkit() is st.session_state["toolkit"]
        """)

        # Act
        at.run()

        # Assert
        with check:
            assert not at.exception, f"Unexpected Streamlit exception: {at.exception}"
        with check:
            assert at.session_state["_toolkit_type"] == "DataFrameToolkit", (
                "get_toolkit() must return a DataFrameToolkit"
            )
        with check:
            assert at.session_state["_same_object"] is True, (
                "get_toolkit() should return the same object stored in session state"
            )


# ---------------------------------------------------------------------------
# mark_agent_stale / is_agent_stale
# ---------------------------------------------------------------------------


class TestAgentStaleness:
    """Tests for mark_agent_stale() and is_agent_stale()."""

    def test_mark_agent_stale_and_check_returns_true(self) -> None:
        """After mark_agent_stale(), is_agent_stale() must return True."""
        # Arrange — start with a non-stale state
        at = AppTest.from_string("""
            import streamlit as st
            from dfchat.state import init_session_state, mark_agent_stale, is_agent_stale
            init_session_state()
            st.session_state["agent_stale"] = False
            mark_agent_stale()
            st.session_state["_is_stale"] = is_agent_stale()
        """)

        # Act
        at.run()

        # Assert
        with check:
            assert not at.exception, f"Unexpected Streamlit exception: {at.exception}"
        with check:
            assert at.session_state["_is_stale"] is True, "is_agent_stale() should return True after mark_agent_stale()"
        with check:
            assert at.session_state["agent_stale"] is True, "agent_stale key should be True in session state"

    def test_is_agent_stale_returns_false_when_not_stale(self) -> None:
        """is_agent_stale() should return False when agent_stale is False."""
        # Arrange
        at = AppTest.from_string("""
            import streamlit as st
            from dfchat.state import init_session_state, is_agent_stale
            init_session_state()
            st.session_state["agent_stale"] = False
            st.session_state["_is_stale"] = is_agent_stale()
        """)

        # Act
        at.run()

        # Assert
        with check:
            assert not at.exception, f"Unexpected Streamlit exception: {at.exception}"
        with check:
            assert at.session_state["_is_stale"] is False, (
                "is_agent_stale() should reflect False when state key is False"
            )

    def test_is_agent_stale_returns_true_after_init(self) -> None:
        """init_session_state sets agent_stale to True, so is_agent_stale() should reflect that."""
        # Arrange — no changes after init
        at = AppTest.from_string("""
            import streamlit as st
            from dfchat.state import init_session_state, is_agent_stale
            init_session_state()
            st.session_state["_is_stale"] = is_agent_stale()
        """)

        # Act
        at.run()

        # Assert
        with check:
            assert not at.exception, f"Unexpected Streamlit exception: {at.exception}"
        with check:
            assert at.session_state["_is_stale"] is True, "agent_stale defaults to True after init"


# ---------------------------------------------------------------------------
# set_selected_df / get_selected_df
# ---------------------------------------------------------------------------


class TestSelectedDf:
    """Tests for set_selected_df() and get_selected_df()."""

    def test_set_and_get_selected_df_round_trip(self) -> None:
        """set_selected_df then get_selected_df should return the same name."""
        # Arrange
        at = AppTest.from_string("""
            import streamlit as st
            from dfchat.state import init_session_state, set_selected_df, get_selected_df
            init_session_state()
            set_selected_df("sales_2024")
            st.session_state["_retrieved_name"] = get_selected_df()
        """)

        # Act
        at.run()

        # Assert
        with check:
            assert not at.exception, f"Unexpected Streamlit exception: {at.exception}"
        with check:
            assert at.session_state["_retrieved_name"] == "sales_2024", (
                "get_selected_df() should return the name set by set_selected_df()"
            )
        with check:
            assert at.session_state["selected_df"] == "sales_2024", "Session state should reflect the new name"

    def test_set_selected_df_none_clears_selection(self) -> None:
        """Setting selected_df to None should clear a previously set name."""
        # Arrange — set a name then clear it in the same script
        at = AppTest.from_string("""
            import streamlit as st
            from dfchat.state import init_session_state, set_selected_df, get_selected_df
            init_session_state()
            set_selected_df("orders")
            set_selected_df(None)
            st.session_state["_retrieved_name"] = get_selected_df()
        """)

        # Act
        at.run()

        # Assert
        with check:
            assert not at.exception, f"Unexpected Streamlit exception: {at.exception}"
        with check:
            assert at.session_state["_retrieved_name"] is None, "get_selected_df() should return None after clearing"
        with check:
            assert at.session_state["selected_df"] is None, "Session state should hold None after clearing"

    def test_get_selected_df_returns_none_after_init(self) -> None:
        """get_selected_df() should return None immediately after init with no selection set."""
        # Arrange
        at = AppTest.from_string("""
            import streamlit as st
            from dfchat.state import init_session_state, get_selected_df
            init_session_state()
            st.session_state["_retrieved_name"] = get_selected_df()
        """)

        # Act
        at.run()

        # Assert
        with check:
            assert not at.exception, f"Unexpected Streamlit exception: {at.exception}"
        with check:
            assert at.session_state["_retrieved_name"] is None, (
                "selected_df should be None before any selection is made"
            )


# ---------------------------------------------------------------------------
# set_recommended_df / get_recommended_df
# ---------------------------------------------------------------------------


class TestRecommendedDf:
    """Tests for set_recommended_df() and get_recommended_df()."""

    def test_set_and_get_recommended_df_round_trip(self) -> None:
        """set_recommended_df then get_recommended_df should return the same name."""
        # Arrange
        at = AppTest.from_string("""
            import streamlit as st
            from dfchat.state import init_session_state, set_recommended_df, get_recommended_df
            init_session_state()
            set_recommended_df("customers_q1")
            st.session_state["_retrieved_name"] = get_recommended_df()
        """)

        # Act
        at.run()

        # Assert
        with check:
            assert not at.exception, f"Unexpected Streamlit exception: {at.exception}"
        with check:
            assert at.session_state["_retrieved_name"] == "customers_q1", (
                "get_recommended_df() should return the name set by set_recommended_df()"
            )
        with check:
            assert at.session_state["recommended_df"] == "customers_q1", (
                "Session state should reflect the recommendation"
            )

    def test_set_recommended_df_none_clears_recommendation(self) -> None:
        """Setting recommended_df to None should clear a previously set recommendation."""
        # Arrange — set a recommendation then clear it in the same script
        at = AppTest.from_string("""
            import streamlit as st
            from dfchat.state import init_session_state, set_recommended_df, get_recommended_df
            init_session_state()
            set_recommended_df("inventory")
            set_recommended_df(None)
            st.session_state["_retrieved_name"] = get_recommended_df()
        """)

        # Act
        at.run()

        # Assert
        with check:
            assert not at.exception, f"Unexpected Streamlit exception: {at.exception}"
        with check:
            assert at.session_state["_retrieved_name"] is None, "get_recommended_df() should return None after clearing"
        with check:
            assert at.session_state["recommended_df"] is None, "Session state should hold None after clearing"

    def test_get_recommended_df_returns_none_after_init(self) -> None:
        """get_recommended_df() should return None immediately after init."""
        # Arrange
        at = AppTest.from_string("""
            import streamlit as st
            from dfchat.state import init_session_state, get_recommended_df
            init_session_state()
            st.session_state["_retrieved_name"] = get_recommended_df()
        """)

        # Act
        at.run()

        # Assert
        with check:
            assert not at.exception, f"Unexpected Streamlit exception: {at.exception}"
        with check:
            assert at.session_state["_retrieved_name"] is None, (
                "recommended_df should be None before any recommendation is set"
            )


# ---------------------------------------------------------------------------
# get_chat_history / append_chat_message
# ---------------------------------------------------------------------------


class TestChatHistory:
    """Tests for get_chat_history() and append_chat_message()."""

    def test_get_chat_history_empty_initially(self) -> None:
        """get_chat_history() should return an empty list right after init."""
        # Arrange
        at = AppTest.from_string("""
            import streamlit as st
            from dfchat.state import init_session_state, get_chat_history
            init_session_state()
            st.session_state["_history"] = get_chat_history()
        """)

        # Act
        at.run()

        # Assert
        with check:
            assert not at.exception, f"Unexpected Streamlit exception: {at.exception}"
        with check:
            assert at.session_state["_history"] == [], "chat_history should be empty immediately after init"

    @pytest.mark.parametrize(
        ("role", "content", "expected_type_name"),
        [
            ("user", "What is the average sales by region?", "HumanMessage"),
            ("assistant", "I found 42 rows matching your filter.", "AIMessage"),
        ],
    )
    def test_append_chat_message_single_message_stored_correctly(
        self, role: str, content: str, expected_type_name: str
    ) -> None:
        """A single appended message should be a BaseMessage with the correct type and content.

        Args:
            role (str): The role string to pass to append_chat_message.
            content (str): The content string to pass to append_chat_message.
            expected_type_name (str): The expected class name of the stored message object.
        """
        # Arrange
        at = AppTest.from_string(f"""
            import streamlit as st
            from dfchat.state import init_session_state, append_chat_message, get_chat_history
            init_session_state()
            append_chat_message("{role}", "{content}")
            history = get_chat_history()
            st.session_state["_history_len"] = len(history)
            st.session_state["_msg_type"] = type(history[0]).__name__
            st.session_state["_msg_content"] = history[0].content
        """)

        # Act
        at.run()

        # Assert
        with check:
            assert not at.exception, f"Unexpected Streamlit exception: {at.exception}"
        with check:
            assert at.session_state["_history_len"] == 1, "History should contain exactly 1 message"
        with check:
            assert at.session_state["_msg_type"] == expected_type_name, (
                f"Message type should be {expected_type_name} for role '{role}'"
            )
        with check:
            assert at.session_state["_msg_content"] == content, "Content should match the appended message"

    def test_append_multiple_chat_messages_preserves_order_and_content(self) -> None:
        """Multiple appended messages should appear in insertion order with correct types and content."""
        # Arrange
        at = AppTest.from_string("""
            import streamlit as st
            from dfchat.state import init_session_state, append_chat_message, get_chat_history
            init_session_state()
            append_chat_message("user", "Show me top 10 customers.")
            append_chat_message("assistant", "Here are the top 10 customers by revenue.")
            history = get_chat_history()
            st.session_state["_history_len"] = len(history)
            st.session_state["_msg0_type"] = type(history[0]).__name__
            st.session_state["_msg0_content"] = history[0].content
            st.session_state["_msg1_type"] = type(history[1]).__name__
            st.session_state["_msg1_content"] = history[1].content
        """)

        # Act
        at.run()

        # Assert
        with check:
            assert not at.exception, f"Unexpected Streamlit exception: {at.exception}"
        with check:
            assert at.session_state["_history_len"] == 2, "History should contain exactly 2 messages"
        with check:
            assert at.session_state["_msg0_type"] == "HumanMessage", (
                "First message should be a HumanMessage for role 'user'"
            )
        with check:
            assert at.session_state["_msg0_content"] == "Show me top 10 customers.", (
                "First message content should match the user input"
            )
        with check:
            assert at.session_state["_msg1_type"] == "AIMessage", (
                "Second message should be an AIMessage for role 'assistant'"
            )
        with check:
            assert at.session_state["_msg1_content"] == "Here are the top 10 customers by revenue.", (
                "Second message content should match the assistant response"
            )

    def test_append_chat_message_invalid_role_raises_value_error(self) -> None:
        """append_chat_message with an unrecognised role should raise ValueError."""
        # Arrange
        at = AppTest.from_string("""
            import streamlit as st
            from dfchat.state import init_session_state, append_chat_message
            init_session_state()
            append_chat_message("system", "You are a helpful assistant.")
        """)

        # Act
        at.run()

        # Assert — AppTest wraps script exceptions in an ElementList; inspect its message
        # to confirm the ValueError was raised with the expected text.
        assert at.exception is not None, "An exception should have been raised for an invalid role"
        exception_list = list(at.exception)
        assert len(exception_list) == 1, f"Expected exactly one exception, got {len(exception_list)}"
        assert "Invalid role" in exception_list[0].message, (
            f"Expected 'Invalid role' in exception message, got: {exception_list[0].message}"
        )


# Suppress the unused-import warning: DataFrameToolkit is used only in
# isinstance() assertions that read from at.session_state after deserialization.
_ = pytest.importorskip
