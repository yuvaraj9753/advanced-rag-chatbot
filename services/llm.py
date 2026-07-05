import os
import requests
from dotenv import load_dotenv

load_dotenv()


def generate_answer(question, context, memory_text=""):

    prompt = f"""
You are an intelligent AI assistant.

Follow these rules:

1. If the question is about the uploaded document, answer using the document context.
2. If the answer can be reasonably inferred from the context (for example, "Where is the train going?" from boarding and destination), do so.
3. If the question is a normal conversational message (like "Hi", "Hello", "Thanks", "Okay", "Bye"), respond naturally.
4. If the question is general knowledge and is NOT related to the uploaded document, answer using your general knowledge.
5. If the user is clearly asking about the uploaded document but the information is not present, reply:
   "I couldn't find this information in the uploaded document."

Context:
{context}

Question:
{question}

CHAT HISTORY:
{memory_text}
"""

    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=data
    )

    return response.json()["choices"][0]["message"]["content"]