import os
import time
import streamlit as st
from services.rag_pipeline import RAGSystem

st.set_page_config(
    page_title="Advanced RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

# -------------------- CSS --------------------

st.markdown("""
<style>

.chat-user{
    background:#DCF8C6;
    padding:10px;
    border-radius:10px;
    margin-bottom:10px;
}

.chat-bot{
    background:#F1F0F0;
    padding:10px;
    border-radius:10px;
    margin-bottom:10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------- Session --------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "summary" not in st.session_state:
    st.session_state.summary = None

# -------------------- Title --------------------

st.title("📚 Advanced RAG Chatbot")

st.caption(
    "Upload a PDF and chat with it using Retrieval-Augmented Generation (RAG)"
)

# -------------------- Sidebar --------------------

with st.sidebar:

    st.header("📂 Document Manager")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type="pdf"
    )

    if uploaded_file:

        st.success("✅ Document Uploaded")

        st.write(f"**Name :** {uploaded_file.name}")

        st.write(
            f"**Size :** {uploaded_file.size/1024:.2f} KB"
        )

    st.divider()

    if "rag" in st.session_state:

        rag = st.session_state.rag

        st.subheader("📊 Document Statistics")

        st.write(f"📄 Pages : {rag.total_pages}")

        st.write(f"🧩 Chunks : {rag.total_chunks}")

        st.write(
            f"🧠 Embedding : {rag.embedding_model}"
        )

        st.write("🗂 Vector DB : FAISS")

        st.write("🤖 LLM : Llama-3.3-70B")

        if hasattr(rag, "processing_time"):

            st.write(
                f"⏱ Processing : {rag.processing_time:.2f} sec"
            )

        st.success("✅ Ready")

    st.divider()

    if st.button("🧹 Clear Chat"):

        st.session_state.messages = []

        st.session_state.summary = None

        if "rag" in st.session_state:
            del st.session_state.rag

        st.rerun()

# -------------------- Load PDF --------------------

if uploaded_file and "rag" not in st.session_state:

    os.makedirs("uploads", exist_ok=True)

    pdf_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    progress = st.progress(0)

    status = st.empty()

    start = time.time()

    def update_progress(message, percent):

        status.info(message)

        progress.progress(percent)

    rag = RAGSystem()

    rag.build_index(
        pdf_path,
        progress_callback=update_progress
    )

    rag.processing_time = time.time() - start

    st.session_state.rag = rag

    st.session_state.summary = None

    progress.progress(100)

    status.success("✅ RAG System Ready")

    st.success(
        f"Document processed successfully in "
        f"{rag.processing_time:.2f} seconds."
    )

    st.rerun()

# -------------------- Tabs --------------------

tab_chat, tab_summary, tab_chunks = st.tabs(
    [
        "💬 Chat",
        "📄 Summary",
        "🧩 Chunks"
    ]
)
# -------------------- Chat Tab --------------------

with tab_chat:

    # Chat History
    for msg in st.session_state.messages:

        if msg["role"] == "user":

            st.markdown(
                f"<div class='chat-user'>🧑 {msg['content']}</div>",
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"<div class='chat-bot'>🤖 {msg['content']}</div>",
                unsafe_allow_html=True
            )

    # Chat Input
    question = st.chat_input(
        "Ask a question about your document..."
    )

    if question and uploaded_file:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.spinner("Thinking..."):

            answer = st.session_state.rag.ask(
                question,
                chat_history=st.session_state.messages[-15:]
            )

        st.session_state.messages.append(
            {
                "role": "bot",
                "content": answer
            }
        )

        st.rerun()

    elif question and not uploaded_file:

        st.warning(
            "Please upload a PDF first."
        )

# -------------------- Summary Tab --------------------

with tab_summary:

    st.subheader("📄 AI Document Summary")

    if "rag" not in st.session_state:

        st.info("Upload a PDF to generate a summary.")

    else:

        if st.button("📄 Generate Summary"):

            with st.spinner("Generating Summary..."):

                st.session_state.summary = (
                    st.session_state.rag.get_summary()
                )

        if st.session_state.summary:

            st.markdown(st.session_state.summary)

# -------------------- Chunks Tab --------------------

with tab_chunks:

    st.subheader("🧩 Chunk Explorer")

    if "rag" not in st.session_state:

        st.info("Upload a PDF to view chunks.")

    else:

        chunks = st.session_state.rag.get_chunks()

        st.write(f"**Total Chunks : {len(chunks)}**")

        for chunk in chunks:

            with st.expander(
                f"Chunk {chunk['chunk']} | Page {chunk['page']}"
            ):

                col1, col2 = st.columns(2)

                with col1:
                    st.write(
                        f"**Characters :** {chunk['characters']}"
                    )

                with col2:
                    st.write(
                        f"**Words :** {chunk['words']}"
                    )

                st.text_area(
                    label="Chunk Content",
                    value=chunk["text"],
                    height=180,
                    disabled=True,
                    key=f"chunk_{chunk['chunk']}"
                )