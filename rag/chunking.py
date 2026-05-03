from typing import List, Dict
import re


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> List[str]:
    text = clean_text(text)
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def build_chunks(
    paragraphs: List[str],
    chunk_size_chars: int = 1200,
    overlap_chars: int = 200,
) -> List[Dict]:
    chunks = []
    current = ""
    start_para = 0
    chunk_id = 0

    for i, para in enumerate(paragraphs):
        candidate = current + "\n\n" + para if current else para

        if len(candidate) <= chunk_size_chars:
            if not current:
                start_para = i
            current = candidate
        else:
            if current:
                chunks.append(
                    {
                        "chunk_id": f"chunk_{chunk_id:05d}",
                        "text": current,
                        "start_para": start_para,
                        "end_para": i - 1,
                    }
                )
                chunk_id += 1

            overlap = current[-overlap_chars:] if current else ""
            current = overlap + "\n\n" + para if overlap else para
            start_para = i

    if current:
        chunks.append(
            {
                "chunk_id": f"chunk_{chunk_id:05d}",
                "text": current,
                "start_para": start_para,
                "end_para": len(paragraphs) - 1,
            }
        )

    return chunks