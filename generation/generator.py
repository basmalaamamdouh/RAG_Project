import os
from groq import Groq
from dotenv import load_dotenv

# =========================
# Load environment variables
# =========================
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# =========================
# 1. Build prompt (CLEAN + FINAL)
# =========================
def build_prompt(query, retrieved_chunks, mode="beginner"):
    context = "\n\n".join(retrieved_chunks)

    if mode == "beginner":
        instruction = "Explain in a very simple way with intuition. Avoid heavy math."
    elif mode == "intermediate":
        instruction = "Explain clearly with moderate technical depth."
    else:
        instruction = "Explain in a deep technical and mathematical way."

    prompt = f"""
You are an AI tutor specialized in Machine Learning.

Context:
{context}

Question:
{query}

Instructions:
{instruction}

Format:
- Simple Explanation
- Deeper Explanation
- Example (if needed)
- Notes (optional)

Answer:
"""
    return prompt


# =========================
# 2. Generate answer (LLM call)
# =========================
def generate_answer(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content