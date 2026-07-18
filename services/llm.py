import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api.groq.com/openai/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
    "Content-Type": "application/json"
}


def call_llm(prompt, temperature=0.2):

    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature
    }

    response = requests.post(
        API_URL,
        headers=HEADERS,
        json=data
    )

    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


# --------------------------------------------------
# Main Question Answering
# --------------------------------------------------

def generate_answer(question, context, memory_text=""):

    prompt = f"""
You are an Advanced RAG Document Assistant.

Your task is to answer questions ONLY using the retrieved document context.

Instructions:

1. Read the complete document context carefully.
2. Use ONLY the document context.
3. If multiple items exist (projects, skills, names, experience, dates etc.), return ALL relevant items.
4. Combine information from different chunks whenever required.
5. Never hallucinate.
6. If the information is unavailable, reply exactly:
"I couldn't find this information in the uploaded document."
7. Keep answers clean and well structured.
8. Use bullet points whenever suitable.

Document Context:
{context}

Previous Conversation:
{memory_text}

User Question:
{question}

Answer:
"""

    return call_llm(prompt, temperature=0.2)


# --------------------------------------------------
# AI Document Summary
# --------------------------------------------------

def generate_summary(context):

    prompt = f"""
You are an AI Document Summarizer.

Your job is to summarize ONLY the uploaded document.

Rules:

1. Read the complete document carefully.
2. Do NOT add any external knowledge.
3. Do NOT hallucinate.
4. Keep the summary concise but informative.
5. Return the summary exactly in this format.

# 📄 Overview

Write 2-3 sentences explaining what this document is about.

# 🔑 Key Points

- Point 1
- Point 2
- Point 3
- Point 4
- Point 5

# 📌 Important Information

Mention important names, dates, places, numbers or facts if available.

# ✅ Conclusion

Write 2-3 sentences summarizing the document.

Document Context:

{context}
"""

    return call_llm(prompt, temperature=0.2)