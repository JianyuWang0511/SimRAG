import os
import time
from typing import List, Dict, Tuple

import numpy as np

from rag.index_store import load_json, save_json, load_vectors, save_vectors, cosine_search


class ExperienceStore:
    def __init__(self, store_path: str, vector_path: str, embedder):
        self.store_path = store_path
        self.vector_path = vector_path
        self.embedder = embedder

        self.experiences: List[Dict] = []
        self.vectors = None

        self._load()

    def _load(self):
        if os.path.exists(self.store_path):
            self.experiences = load_json(self.store_path)
        else:
            self.experiences = []

        if os.path.exists(self.vector_path):
            self.vectors = load_vectors(self.vector_path)
        else:
            self.vectors = None

    def save(self):
        save_json(self.store_path, self.experiences)

        if self.vectors is not None:
            save_vectors(self.vector_path, self.vectors)

    def search(self, question: str, top_k: int) -> List[Tuple[Dict, float]]:
        if not self.experiences or self.vectors is None:
            return []

        q_vec = self.embedder.embed_one(question)
        results = cosine_search(q_vec, self.vectors, top_k)

        return [(self.experiences[idx], score) for idx, score in results]

    def add_experience(
        self,
        question: str,
        answer: str,
        retrieved_chunk_ids: List[str],
        cited_chunk_ids: List[str],
        feedback_score: float = 0.5,
    ):
        exp = {
            "experience_id": f"exp_{int(time.time() * 1000)}",
            "question": question,
            "answer": answer,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "cited_chunk_ids": cited_chunk_ids,
            "feedback_score": feedback_score,
            "created_at": time.time(),
        }

        q_vec = self.embedder.embed_one(question).reshape(1, -1)

        if self.vectors is None:
            self.vectors = q_vec
        else:
            self.vectors = np.vstack([self.vectors, q_vec])

        self.experiences.append(exp)
        self.save()

        return exp

    def update_feedback(self, experience_id: str, feedback_score: float) -> bool:
        for exp in self.experiences:
            if exp["experience_id"] == experience_id:
                exp["feedback_score"] = feedback_score
                self.save()
                return True
        return False