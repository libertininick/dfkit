"""Session state management for the dfchat application.

This module provides a thin, idempotent abstraction over ``st.session_state``
so that the rest of the application accesses session state through typed
functions rather than raw dict-style key lookups.
"""

from __future__ import annotations

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from dfkit import DataFrameToolkit


def init_session_state() -> None:
    """Initialize all session state keys if not already present.

    Idempotent: keys that already exist in ``st.session_state`` are left
    unchanged, so this function is safe to call on every page render.

    Creates:
        toolkit (DataFrameToolkit): Shared toolkit instance.
        chat_history (list[BaseMessage]): Conversation history; each entry is a
            LangChain ``BaseMessage`` (``HumanMessage`` or ``AIMessage``).
        agent (CompiledStateGraph | None): Compiled LangGraph agent, or
            ``None`` until the agent has been built.
        agent_stale (bool): Whether the agent needs to be rebuilt before the
            next invocation.
        selected_df (str | None): Name of the currently selected DataFrame, or
            ``None`` if none is selected.
        recommended_df (str | None): Name of the recommended DataFrame, or
            ``None`` if no recommendation is active.
    """
    _build_default_session_state()


def get_toolkit() -> DataFrameToolkit:
    """Return the shared toolkit instance from session state.

    Returns:
        DataFrameToolkit: The active toolkit for this session.
    """
    toolkit: DataFrameToolkit = st.session_state["toolkit"]
    return toolkit


def mark_agent_stale() -> None:
    """Mark the agent as stale so it is rebuilt before the next invocation."""
    st.session_state["agent_stale"] = True


def is_agent_stale() -> bool:
    """Return whether the agent needs to be rebuilt.

    Returns:
        bool: ``True`` if the agent is stale and must be rebuilt.
    """
    is_stale: bool = st.session_state["agent_stale"]
    return is_stale


def set_selected_df(name: str | None) -> None:
    """Update the currently selected DataFrame name.

    Args:
        name (str | None): The DataFrame name to select, or ``None`` to clear
            the selection.
    """
    st.session_state["selected_df"] = name


def get_selected_df() -> str | None:
    """Return the currently selected DataFrame name.

    Returns:
        str | None: The selected DataFrame name, or ``None`` if nothing is
            selected.
    """
    selected: str | None = st.session_state["selected_df"]
    return selected


def set_recommended_df(name: str | None) -> None:
    """Update the recommended DataFrame name.

    Args:
        name (str | None): The DataFrame name to recommend, or ``None`` to
            clear the recommendation.
    """
    st.session_state["recommended_df"] = name


def get_recommended_df() -> str | None:
    """Return the recommended DataFrame name.

    Returns:
        str | None: The recommended DataFrame name, or ``None`` if there is no
            active recommendation.
    """
    recommended: str | None = st.session_state["recommended_df"]
    return recommended


def get_chat_history() -> list[BaseMessage]:
    """Return a shallow copy of the chat history list.

    Returns:
        list[BaseMessage]: Copy of LangChain messages (``HumanMessage`` or
            ``AIMessage`` instances).
    """
    chat_history: list[BaseMessage] = st.session_state["chat_history"]
    return list(chat_history)


def append_chat_message(role: str, content: str) -> None:
    """Append a message to the chat history.

    Args:
        role (str): The message author role; must be ``"user"`` or
            ``"assistant"``.
        content (str): The message text.

    Raises:
        ValueError: If ``role`` is not ``"user"`` or ``"assistant"``.
    """
    if role == "user":
        message: BaseMessage = HumanMessage(content=content)
    elif role == "assistant":
        message = AIMessage(content=content)
    else:
        valid_roles: tuple[str, ...] = ("user", "assistant")
        raise ValueError(f"Invalid role {role!r}; expected one of {valid_roles}")
    st.session_state["chat_history"].append(message)


def _init_toolkit_state() -> None:
    """Initialize toolkit and chat history keys if absent."""
    if "toolkit" not in st.session_state:
        st.session_state["toolkit"] = DataFrameToolkit()
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "agent" not in st.session_state:
        st.session_state["agent"] = None


def _init_ui_state() -> None:
    """Initialize agent and DataFrame selection keys if absent."""
    if "agent_stale" not in st.session_state:
        st.session_state["agent_stale"] = True
    if "selected_df" not in st.session_state:
        st.session_state["selected_df"] = None
    if "recommended_df" not in st.session_state:
        st.session_state["recommended_df"] = None


def _build_default_session_state() -> None:
    """Initialize all missing session state keys with their default values.

    Delegates to focused helpers so each stays within the complexity limit.
    Expensive objects such as ``DataFrameToolkit`` are constructed lazily and
    only when their key is absent from ``st.session_state``.
    """
    _init_toolkit_state()
    _init_ui_state()
