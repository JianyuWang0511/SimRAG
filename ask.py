from __future__ import annotations
import argparse

from rag.config import RAGConfig
from rag.pipeline import NovelRAG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=str, required=True, help="Question to ask about the novel.")
    parser.add_argument("--show_sources", action="store_true", help="Print retrieved chunks.")
    args = parser.parse_args()

    cfg = RAGConfig()
    rag = NovelRAG(cfg)

    answer, retrieved = rag.answer(args.q)
    print("\nANSWER:\n")
    print(answer)

    if args.show_sources:
        print("\n\nSOURCES:\n")
        for r in retrieved:
            print(f"- {r.chunk_id} (score={r.score:.4f}) paras {r.meta.get('start_para_idx')}–{r.meta.get('end_para_idx')}")
            print(r.text[:600].strip() + ("..." if len(r.text) > 600 else ""))
            print("")

if __name__ == "__main__":
    main()