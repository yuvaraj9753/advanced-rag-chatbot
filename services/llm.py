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

Your primary task is to answer questions using ONLY the uploaded document.

Special Instructions:

1. If the user sends only a greeting or a polite conversational message such as:
   "Hi", "Hello", "Hey", "Good Morning", "Good Afternoon",
   "Good Evening", "Thanks", "Thank You", "Bye", "Goodbye",
   "Okay", or "Ok", respond naturally and politely.

2. Do NOT reply with:
   "I couldn't find this information in the uploaded document."
   for greetings, thanks, or farewell messages.

3. Only use the response:
   "I couldn't find this information in the uploaded document."
   when the user is asking for information that should exist in the uploaded document but is not available.

4. Read the complete document context carefully before answering.

5. Use ONLY the information available in the document context.

6. If multiple relevant items exist (such as projects, skills, names, dates, experience, policies, or lists), include ALL relevant information.

7. Combine information from multiple retrieved chunks whenever necessary.

8. Never hallucinate, invent, or assume information.

9. Keep the answer clear, accurate, and well-structured.

10. Use bullet points or numbered lists whenever appropriate.

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