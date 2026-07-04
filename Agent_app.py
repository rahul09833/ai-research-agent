

import os
import streamlit as st
from Agent_core import run_smart_agent

st.set_page_config(page_title="AI Research Agent", page_icon="🤖")
st.title("🤖 AI Research Agent")
st.caption("Ask for anything. Simple asks get answered directly, complex ones get broken into steps automatically.")

if not os.environ.get("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY not found. Add it to a .env file in this folder and restart the app.")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"/"assistant", "content": str}

# Render past conversation
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])

# New input
user_input = st.chat_input("What do you want the agent to do?")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        activity_box = st.status("Working...", expanded=True)
        log_placeholder = activity_box.empty()

        answer = run_smart_agent(
            user_input,
            chat_history=st.session_state.chat_history[:-1],  # exclude the message just added
            log_placeholder=log_placeholder,
        )
        activity_box.update(label="Done", state="complete", expanded=False)
        st.write(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})

if st.session_state.chat_history:
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()

st.divider()
st.caption("Logs are also saved to agent.log in this folder for debugging.")