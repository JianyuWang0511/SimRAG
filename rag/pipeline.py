from rag.config import RAGConfig
from rag.embedder import OpenAIEmbedder
from rag.retrieval import retrieve_chunks
from rag.generator import generate_answer
from rag.experience import ExperienceStore


class RAGPipeline:
    def __init__(self):
        self.cfg = RAGConfig()
        self.embedder = OpenAIEmbedder(self.cfg.embedding_model)
        self.exp_store = ExperienceStore(
            self.cfg.experience_store_path,
            self.cfg.experience_index_path,
            self.embedder,
        )

    def run(self, question: str):
        # Step 1: retrieve similar past questions
        neighbors = self.exp_store.search(question, top_k=self.cfg.top_k_experiences)

        # Step 2: retrieve chunks (your fusion + citation logic goes here)
        chunks = retrieve_chunks(
            question,
            neighbors,
            self.embedder,
            self.cfg,
        )

        # Step 3: generate answer
        answer = generate_answer(question, chunks, self.cfg)

        # Step 4: store experience
        self.exp_store.add(
            question=question,
            answer=answer,
            chunks=[c["chunk_id"] for c in chunks],
        )

        return answer, chunks