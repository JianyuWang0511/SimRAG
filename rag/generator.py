import os
import re
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def build_context(chunks: List[Dict]) -> str:
    context_blocks = []

    for chunk in chunks:
        block = (
            f"[{chunk['chunk_id']} | paragraphs "
            f"{chunk['start_para']}-{chunk['end_para']}]\n"
            f"{chunk['text']}"
        )
        context_blocks.append(block)

    return "\n\n---\n\n".join(context_blocks)


def generate_answer(question: str, chunks: List[Dict], cfg) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

    client = OpenAI(api_key=api_key)

    context = build_context(chunks)

    system_prompt = """
You are a careful RAG assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. If the context is insufficient, say you do not know.
3. Cite the chunk IDs you used.
4. Use citations in this format: (chunk_00001).
5. Do not cite chunks that are not in the provided context.
"""

    user_prompt = f"""
CONTEXT:

{context}

QUESTION:

{question}

Now answer the question with chunk citations.
"""

    response = client.chat.completions.create(
        model=cfg.chat_model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=cfg.temperature,
    )

    return response.choices[0].message.content


def extract_cited_chunk_ids(answer: str) -> List[str]:
    """
    Finds citations like (chunk_00001).
    """
    cited = re.findall(r"chunk_\d{5}", answer)
    return sorted(set(cited))