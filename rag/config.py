from dataclasses import dataclass


@dataclass
class RAGConfig:
    novel_path: str = "data/novel.txt"

    chunk_index_path: str = "data/chunk_index.faiss"
    chunk_store_path: str = "data/chunks.json"

    experience_index_path: str = "data/experience_index.faiss"
    experience_store_path: str = "data/experiences.json"

    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4.1-mini"

    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 200

    top_k_chunks: int = 5
    top_k_experiences: int = 3
    candidate_k_per_query: int = 10

    neighbor_weight: float = 0.4
    citation_boost_weight: float = 0.6

    temperature: float = 0.2