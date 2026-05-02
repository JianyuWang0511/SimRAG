import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class OpenAIEmbedder:
    def __init__(self, model: str):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Check your .env file.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_one(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        vec = np.array(response.data[0].embedding, dtype="float32")
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec

    def embed_many(self, texts):
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        vectors = []
        for item in response.data:
            vec = np.array(item.embedding, dtype="float32")
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            vectors.append(vec)
        return np.vstack(vectors).astype("float32")