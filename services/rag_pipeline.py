import os

from services.pdf_loader import load_pdf
from services.text_splitter import split_documents
from services.embeddings import get_embeddings
from services.vector_store import (
    load_or_create_vectorstore,
    retrieve_documents
)
from services.llm import generate_answer


class RAGSystem:

    def __init__(self):

        self.vectorstore = None
        self.total_pages = 0
        self.total_chunks = 0
        self.embedding_model = "all-MiniLM-L6-v2"
        self.processing_time = 0

    def build_index(self, pdf_path, progress_callback=None):

        self.pdf_path = pdf_path

        # -------------------- Step 1 --------------------
        if progress_callback:
            progress_callback("📄 Loading PDF...", 20)

        documents = load_pdf(pdf_path)

        self.total_pages = len(documents)

        # -------------------- Step 2 --------------------
        if progress_callback:
            progress_callback("✂️ Splitting document into chunks...", 40)

        docs = split_documents(documents)

        self.total_chunks = len(docs)

        # -------------------- Step 3 --------------------
        if progress_callback:
            progress_callback("🧠 Loading embedding model...", 60)

        embeddings = get_embeddings()

        # -------------------- Step 4 --------------------
        if progress_callback:
            progress_callback("🗂 Creating FAISS Vector Store...", 85)

        self.vectorstore = load_or_create_vectorstore(
            pdf_path,
            docs,
            embeddings
        )

        # -------------------- Step 5 --------------------
        if progress_callback:
            progress_callback("✅ RAG System Ready!", 100)

    def ask(self, question, chat_history=None):
        memory_text = ""
        if chat_history:
            memory_text = "\n".join(
                f"{msg['role']}: {msg['content']}"
                for msg in chat_history
            )

        docs_and_scores = retrieve_documents(
            self.vectorstore,
            question
        )

        if not docs_and_scores:

            return (
                "I couldn't find any relevant information "
                "in the uploaded document."
            )

        filtered_docs = [
            doc for doc, score in docs_and_scores
        ]

        context = "\n\n".join(
            doc.page_content
            for doc in filtered_docs
        )

        answer = generate_answer(
            question,
            context
        )

        sources = []

        for doc in filtered_docs:

            page = doc.metadata.get(
                "page",
                "Unknown"
            )

            source = os.path.basename(
                doc.metadata.get(
                    "source",
                    "Document"
                )
            )

            if page != "Unknown":

                sources.append(
                    f"{source} (Page {page + 1})"
                )

            else:

                sources.append(source)

        unique_sources = list(
            dict.fromkeys(sources)
        )

        answer += "\n\n---\n📄 **Sources:**\n"

        for src in unique_sources:
            answer += f"- {src}\n"

        return answer