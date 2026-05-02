import json
import os
from typing import List, Dict, Tuple

import faiss
import numpy as np


def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_faiss_index(index: faiss.Index, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)


def load_faiss_index(path: str) -> faiss.Index:
    return faiss.read_index(path)


def search_index(
    index: faiss.Index,
    query_vector: np.ndarray,
    top_k: int,
) -> List[Tuple[int, float]]:
    query_vector = query_vector.reshape(1, -1).astype("float32")
    scores, indices = index.search(query_vector, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx != -1:
            results.append((int(idx), float(score)))
    return results