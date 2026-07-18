import os
import hashlib
from langchain_community.vectorstores import FAISS


def get_pdf_hash(pdf_path):

    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_or_create_vectorstore(pdf_path, docs, embeddings):

    os.makedirs("indexes", exist_ok=True)

    file_hash = get_pdf_hash(pdf_path)

    index_path = os.path.join(
        "indexes",
        file_hash
    )

    if os.path.exists(index_path):

        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        print("✅ Loaded Cached FAISS Index")

    else:

        vectorstore = FAISS.from_documents(
            docs,
            embeddings
        )

        vectorstore.save_local(index_path)

        print("🔥 Created New FAISS Index")

    return vectorstore


def retrieve_documents(vectorstore, question):

    question_lower = question.lower()

    # Dynamic Retrieval
    if any(keyword in question_lower for keyword in [

        "all",
        "list",
        "complete",
        "every",
        "summary",
        "summarize",
        "experience",
        "projects",
        "skills"

    ]):

        k = 10

    else:

        k = 5

    retriever = vectorstore.as_retriever(

        search_type="mmr",

        search_kwargs={

            "k": k,

            "fetch_k": 30,

            "lambda_mult": 0.5

        }

    )

    docs = retriever.invoke(question)

    return [

        (doc, 0.0)

        for doc in docs

    ]