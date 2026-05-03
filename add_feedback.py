import argparse

from rag.config import RAGConfig
from rag.embedder import OpenAIEmbedder
from rag.experience import ExperienceStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument("--score", type=float, required=True)

    args = parser.parse_args()

    if args.score < 0 or args.score > 1:
        raise ValueError("Feedback score must be between 0 and 1.")

    cfg = RAGConfig()
    embedder = OpenAIEmbedder(cfg.embedding_model)

    store = ExperienceStore(
        store_path=cfg.experience_store_path,
        vector_path=cfg.experience_vectors_path,
        embedder=embedder,
    )

    success = store.update_feedback(
        experience_id=args.id,
        feedback_score=args.score,
    )

    if success:
        print(f"Updated feedback for {args.id} to {args.score}")
    else:
        print(f"Experience ID not found: {args.id}")


if __name__ == "__main__":
    main()