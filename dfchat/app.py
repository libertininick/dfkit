"""Entry point for the dfchat application.

Run with: uv run streamlit run dfchat/app.py
"""

import streamlit as st

from dfchat.state import init_session_state


def main() -> None:
    """Run the dfchat application.

    Configures the page, initializes session state, and renders the top-level
    tab layout. Tab content is populated in later phases.
    """
    st.set_page_config(page_title="dfchat", layout="wide")
    init_session_state()

    # Sidebar placeholder (Phase 4)
    with st.sidebar:
        st.header("DataFrames")
        st.info("No DataFrames registered yet.")

    # Tab routing
    data_tab, analysis_tab = st.tabs(["Data Setup", "Analysis"])

    with data_tab:
        st.header("Data Setup")
        st.info("Upload and register DataFrames here. (Coming in Phase 5)")

    with analysis_tab:
        st.header("Analysis")
        st.info("Query your data with natural language. (Coming in Phase 8)")


if __name__ == "__main__":
    main()
