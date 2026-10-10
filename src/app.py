import os
import time
import streamlit as st

from rag import ask_document
from ingest import ingest_document

st.set_page_config(
    page_title="Budhana Tech AI Assistant",
    page_icon="🤖",
    layout="wide",
)

# -------------------- Session State --------------------

if "documents" not in st.session_state:
    st.session_state.documents = {}

if "active_document" not in st.session_state:
    st.session_state.active_document = None

if "chats" not in st.session_state:
    st.session_state.chats = {}

# -------------------- Header --------------------

header, monitoring = st.columns([4, 1])

with header:
    st.title("🤖 Budhana Tech Knowledge Assistant")
    st.caption("Ask questions from approved company documents.")

with monitoring:
    st.markdown("### LangSmith")
    project = os.getenv("LANGCHAIN_PROJECT", "budhana-rag-chatbot")
    st.caption(f"Project: {project}")

    # Set LANGSMITH_PROJECT_URL in .env to your actual project URL.
    project_url = os.getenv("LANGSMITH_PROJECT_URL")
    if project_url:
        st.link_button("View traces ↗", project_url)

st.divider()

# -------------------- Two-Panel Layout --------------------

left_panel, right_panel = st.columns([1, 2.2], gap="large")

# ==================== LEFT PANEL ====================

with left_panel:
    st.subheader("📁 Knowledge Documents")

    uploaded_file = st.file_uploader(
        "Upload a text document",
        type=["txt"],
        help="PDF support can be added later.",
    )

    if uploaded_file is not None:
        if st.button("Upload & Process", use_container_width=True):
            file_name = uploaded_file.name
            document_id = file_name
            text = uploaded_file.getvalue().decode("utf-8-sig")

            if not text.strip():
                st.error("The uploaded file is empty.")
            elif document_id in st.session_state.documents:
                st.warning(
                    "This filename already exists. "
                    "Use a different filename to upload another version."
                )
            else:
                status = st.status(
                    "Processing document...",
                    expanded=True,
                )

                try:
                    status.write("1. Upload done ✓")

                    status.write("2. Chunking document...")
                    result = ingest_document(
                        document_id=document_id,
                        file_name=file_name,
                        text=text,
                    )

                    # The backend should return only after
                    # chunking, embedding and persistence succeed.
                    status.write("3. Chunking done ✓")
                    status.write("4. Vectorization done ✓")
                    status.write("5. Saved to ChromaDB ✓")

                    st.session_state.documents[document_id] = {
                        "name": file_name,
                        "chunks": result.get("chunks", 0),
                        "status": "Ready for chat",
                    }

                    st.session_state.chats.setdefault(document_id, [])
                    st.session_state.active_document = document_id

                    status.update(
                        label="Document ready for chat!",
                        state="complete",
                        expanded=True,
                    )

                    st.rerun()

                except Exception as error:
                    status.update(
                        label="Document processing failed",
                        state="error",
                        expanded=True,
                    )
                    st.error(str(error))

    st.divider()
    st.markdown("**Uploaded documents**")

    if not st.session_state.documents:
        st.info("Upload your first document to begin.")

    for doc_id, doc in st.session_state.documents.items():
        is_active = st.session_state.active_document == doc_id

        if st.button(
            f"{'🟢' if is_active else '📄'} {doc['name']}",
            key=f"select_{doc_id}",
            use_container_width=True,
        ):
            st.session_state.active_document = doc_id
            st.rerun()

        st.caption(f"Status: {doc['status']}")

# ==================== RIGHT PANEL ====================

with right_panel:
    active_id = st.session_state.active_document

    if not active_id:
        st.subheader("💬 Document Chat")
        st.info(
            "Upload a document and select it from the left " "panel to start chatting."
        )

    else:
        doc = st.session_state.documents[active_id]

        st.subheader(f"💬 Chat: {doc['name']}")
        st.caption("Answers are restricted to the selected document.")

        if st.button("Clear this chat"):
            st.session_state.chats[active_id] = []
            st.rerun()

        # Display the selected document's conversation.
        for message in st.session_state.chats[active_id]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message.get("sources"):
                    with st.expander("Sources"):
                        for source in message["sources"]:
                            st.write(source)

                if message.get("elapsed") is not None:
                    st.caption(f"Response time: {message['elapsed']:.2f} s")

        question = st.chat_input(f"Ask about {doc['name']}...")

        if question:
            st.session_state.chats[active_id].append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching this document..."):
                    start_time = time.perf_counter()

                    try:
                        result = ask_document(
                            document_id=active_id,
                            question=question,
                        )

                        elapsed = time.perf_counter() - start_time
                        answer = result["answer"]
                        sources = result.get("sources", [])

                    except Exception:
                        elapsed = time.perf_counter() - start_time
                        answer = (
                            "Sorry, I couldn't process your question. "
                            "Please try again."
                        )
                        sources = []
                        st.error("An error occurred while generating the answer.")

                st.markdown(answer)

                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.write(source)

                st.caption(f"Response time: {elapsed:.2f} s")

            st.session_state.chats[active_id].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "elapsed": elapsed,
                }
            )
