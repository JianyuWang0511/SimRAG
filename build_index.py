from tqdm import tqdm

from rag.config import RAGConfig
from rag.chunking import load_text, split_paragraphs, build_chunks
from rag.embedder import OpenAIEmbedder
from rag.index_store import build_faiss_index, save_faiss_index, save_json


def main():
    cfg = RAGConfig()

    print("Loading novel...")
    text = load_text(cfg.novel_path)

    print("Chunking novel...")
    paragraphs = split_paragraphs(text)
    chunks = build_chunks(
        paragraphs,
        chunk_size_chars=cfg.chunk_size_chars,
        overlap_chars=cfg.chunk_overlap_chars,
    )

    print(f"Created {len(chunks)} chunks.")

    embedder = OpenAIEmbedder(cfg.embedding_model)

    print("Embedding chunks...")
    texts = [c["text"] for c in chunks]

    vectors = []
    batch_size = 32

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        batch_vectors = embedder.embed_many(batch)
        vectors.append(batch_vectors)

    import numpy as np

    vectors = np.vstack(vectors).astype("float32")

    print("Building FAISS index...")
    index = build_faiss_index(vectors)

    print("Saving index and chunks...")
    save_faiss_index(index, cfg.chunk_index_path)
    save_json(cfg.chunk_store_path, chunks)

    print("Done. Chunk index built successfully.")


if __name__ == "__main__":
    main()