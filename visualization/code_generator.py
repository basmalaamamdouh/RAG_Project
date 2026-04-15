import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# =========================
# Build visualization prompt
# =========================
def build_visual_prompt(query):
    return f"""
You are a data visualization expert.

The user asked:
"{query}"

Generate Python code using Plotly to visualize this concept.

Rules:
- ONLY return valid Python code (no explanations)
- Use plotly (plotly.graph_objects or plotly.express)
- The code MUST create a variable called 'fig'
- Do NOT use plt or matplotlib
- Do NOT show the plot (no fig.show())
- Keep it simple and clear

Return only code.
"""


# =========================
# Generate visualization code
# =========================
def generate_visualization_code(query):
    try:
        prompt = build_visual_prompt(query)

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        code = response.choices[0].message.content.strip()

        return code

    except Exception as e:
        print(" Error generating visualization code:", e)
        return None