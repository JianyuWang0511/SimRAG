from collections import defaultdict
from typing import List, Dict, Tuple

from rag.index_store import load_json, load_vectors, cosine_search


def retrieve_with_task_similarity(
    question: str,
    embedder,
    cfg,
    experience_store,
) -> Tuple[List[Dict], List[Tuple[Dict, float]]]:
    """
    Retrieval logic:

    1. Retrieve similar past experiences.
    2. Retrieve chunks for the current question.
    3. Retrieve chunks for similar past questions.
    4. Fuse all retrieval scores.
    5. Boost chunks cited in similar high-quality past experiences.
    """

    chunks = load_json(cfg.chunk_store_path)
    chunk_vectors = load_vectors(cfg.chunk_vectors_path)

    neighbors = experience_store.search(
        question,
        top_k=cfg.top_k_experiences,
    )

    candidate_scores = defaultdict(float)

    # --------------------------------------------------
    # 1. Current question retrieval
    # --------------------------------------------------
    current_vec = embedder.embed_one(question)

    current_results = cosine_search(
        current_vec,
        chunk_vectors,
        top_k=cfg.candidate_k_per_query,
    )

    for chunk_idx, score in current_results:
        candidate_scores[chunk_idx] += score

    # --------------------------------------------------
    # 2. Similar past question retrieval
    # --------------------------------------------------
    for exp, exp_similarity in neighbors:
        neighbor_question = exp["question"]
        feedback_score = exp.get("feedback_score", 0.5)

        neighbor_vec = embedder.embed_one(neighbor_question)

        neighbor_results = cosine_search(
            neighbor_vec,
            chunk_vectors,
            top_k=cfg.candidate_k_per_query,
        )

        neighbor_weight = (
            cfg.neighbor_weight
            * max(0.0, exp_similarity)
            * feedback_score
        )

        for chunk_idx, score in neighbor_results:
            candidate_scores[chunk_idx] += neighbor_weight * score

    # --------------------------------------------------
    # 3. Citation boost
    # --------------------------------------------------
    chunk_id_to_idx = {
        chunk["chunk_id"]: idx for idx, chunk in enumerate(chunks)
    }

    for exp, exp_similarity in neighbors:
        feedback_score = exp.get("feedback_score", 0.5)
        cited_chunk_ids = exp.get("cited_chunk_ids", [])

        boost = (
            cfg.citation_boost_weight
            * max(0.0, exp_similarity)
            * feedback_score
        )

        for chunk_id in cited_chunk_ids:
            if chunk_id in chunk_id_to_idx:
                idx = chunk_id_to_idx[chunk_id]
                candidate_scores[idx] += boost

    # --------------------------------------------------
    # 4. Final ranking
    # --------------------------------------------------
    ranked = sorted(
        candidate_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    top = ranked[: cfg.top_k_chunks]

    retrieved_chunks = []
    for idx, final_score in top:
        chunk = dict(chunks[idx])
        chunk["final_score"] = final_score
        retrieved_chunks.append(chunk)

    return retrieved_chunks, neighbors