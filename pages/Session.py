"""Session persistence: save your work to the repo so it survives new sessions.

The app stores each phase's artifacts on local disk. In a fresh Codespace that
disk starts empty, so this page lets you commit the current session into the
repo (``saved_state/``). A new Codespace clones the repo and restores the work
automatically the first time a page loads it.
"""

from __future__ import annotations

import streamlit as st

from simulation_store import save_session_to_repo, saved_session_files

st.set_page_config(page_title="Session", layout="wide")
st.title("Session: save & restore your work")
st.caption(
    "Save the current session (Phase 1-6 state and portfolios) into the GitHub "
    "repo. A new Codespace then restores it automatically on load, so you do not "
    "lose your work when the container is recreated."
)

st.subheader("Save current session to GitHub")
st.write(
    "Click to copy the session artifacts into the repo and commit/push them. "
    "Restore happens automatically the next time the app loads in a fresh checkout."
)
if st.button("Save session to GitHub", type="primary"):
    with st.spinner("Copying artifacts, committing, and pushing..."):
        message = save_session_to_repo()
    if message.lower().startswith(("saved", "session already")):
        st.success(message)
    else:
        st.warning(message)

st.subheader("Currently saved in the repo")
files = saved_session_files()
if files:
    st.write("These session files are committed and will be restored in a new session:")
    st.table({"Saved file": files})
else:
    st.info("No session saved in the repo yet. Run the phases, then click Save.")

st.divider()
st.caption(
    "Note: session files include the full Monte Carlo scenarios and can be a few "
    "megabytes each, so each save adds a commit. Local disk always takes "
    "precedence over the saved copy, so your latest in-session work is never "
    "overwritten by an older save."
)
