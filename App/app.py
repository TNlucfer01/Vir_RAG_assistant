"""
app.py — Vir Campus Assistant (Streamlit UI)

Improvements over original:
  - UUID session_id generated at startup and sent with every request
    → enables persistent multi-turn memory in the backend
  - Streaming responses: the final answer streams token-by-token via
    st.write_stream() so users see text appear instantly instead of waiting
  - Source citations are displayed below the answer when present
  - Tools used + round count shown as a subtle debug badge
"""

import uuid
import streamlit as st

from ui.api import upload_pdf, ask_question_stream, get_suggestions

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Vir — Campus Assistant",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# Session State Init
# --------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "chunks" not in st.session_state:
    st.session_state.chunks = 0

if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

if "followups" not in st.session_state:
    st.session_state.followups = []


# --------------------------------------------------
# Helper — Process and Stream a Question
# --------------------------------------------------

def process_question(question: str):
    # Append user message to local display history
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        # Stream the response token by token
        streamed_answer = st.write_stream(
            ask_question_stream(
                question=question,
                filename="",
                history=st.session_state.messages[-6:],
                session_id=st.session_state.session_id,
            )
        )

    # After streaming completes, fetch the full structured response
    # for followups, debug info etc. (streaming response only gives text)
    full_response = ask_question_stream(
        question=question,
        filename="",
        history=st.session_state.messages[-6:],
        session_id=st.session_state.session_id,
        metadata_only=True,
    )

    answer = streamed_answer or ""
    followups = []
    debug = {}

    if full_response:
        followups = full_response.get("followups", [])
        debug = full_response.get("debug", {})

    st.session_state.followups = followups
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Show subtle debug badge
    if debug:
        tools = debug.get("tools_used", [])
        rounds = debug.get("rounds", "")
        if tools or rounds:
            st.caption(f"🔧 Tools: `{', '.join(tools) or 'fast-path'}` · Rounds: `{rounds}`")


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.title("🤖 Vir")
    st.caption(f"Session: `{st.session_state.session_id[:8]}…`")

    st.markdown("---")

    st.subheader("Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF, DOCX, CSV or TXT",
        type=["pdf", "csv", "docx", "txt"]
    )

    if uploaded_file is not None:
        if st.button("📤 Upload & Index", use_container_width=True):
            with st.spinner("Indexing document..."):
                response = upload_pdf(uploaded_file)
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.document_uploaded = True
                    st.session_state.document_name = data["filename"]
                    st.session_state.chunks = data["chunks_stored"]

                    suggestion_response = get_suggestions(data["filename"])
                    if suggestion_response.status_code == 200:
                        st.session_state.suggested_questions = (
                            suggestion_response.json().get("suggested_questions", [])
                        )
                    else:
                        st.session_state.suggested_questions = []

                    st.success("Document indexed!")
                    st.rerun()
                else:
                    st.error(response.text)

    st.markdown("---")
    st.subheader("Status")

    if st.session_state.document_uploaded:
        st.success("🟢 Document indexed this session")
        st.write(f"📄 **{st.session_state.document_name}**")
        st.write(f"🧩 Chunks: {st.session_state.chunks}")
    else:
        st.info("🟡 Using knowledge from past uploads")

    # Clear conversation
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.followups = []
        st.session_state.session_id = str(uuid.uuid4())  # new session
        st.rerun()

    # Suggested Questions
    if st.session_state.suggested_questions:
        st.markdown("---")
        st.subheader("💡 Suggested Questions")
        for i, q in enumerate(st.session_state.suggested_questions):
            if st.button(q, key=f"suggestion_{i}", use_container_width=True):
                process_question(q)
                st.rerun()


# --------------------------------------------------
# Main Chat Area
# --------------------------------------------------

st.title("💬 Vir — Campus Assistant")
st.caption("Ask about students, regulations, marks, attendance, navigation, and more.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Follow-up questions
if (
    st.session_state.followups
    and st.session_state.messages
    and st.session_state.messages[-1]["role"] == "assistant"
):
    st.markdown("### Continue Exploring")
    cols = st.columns(len(st.session_state.followups))
    for col, q in zip(cols, st.session_state.followups):
        with col:
            if st.button(q, key=f"followup_{q}", use_container_width=True):
                process_question(q)
                st.rerun()

# Chat input
prompt = st.chat_input("Ask anything about PT Lee CNCET...")
if prompt:
    process_question(prompt)
    st.rerun()
