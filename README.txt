Project Title: Task-Similarity-Aware Retrieval-Augmented Generation (SimRAG)

Author: Jianyu Wang

------------------------------------------------------------
1. Overview
------------------------------------------------------------

This project implements an advanced Retrieval-Augmented Generation (RAG) system 
that answers questions over a long-form narrative corpus (a novel) using 
semantic retrieval and a task-similarity-aware memory mechanism.

Unlike standard RAG systems, which retrieve context based only on the current query, 
this system incorporates past query experience to improve retrieval quality over time.

The system introduces:
- Task similarity routing
- Similarity-weighted multi-query retrieval
- Citation-aware retrieval (experience-based memory)
- Feedback-driven adaptive retrieval (no model retraining required)

------------------------------------------------------------
2. Key Idea
------------------------------------------------------------

Standard RAG:
    Query → Retrieve → Generate

This project:
    Query
    → Retrieve similar past questions (task similarity)
    → Retrieve chunks for current query
    → Retrieve chunks for similar past queries
    → Fuse retrieval scores
    → Boost chunks cited in past successful answers
    → Generate grounded answer
    → Store experience for future queries

The system improves over time using retrieval-level memory instead of model training.

------------------------------------------------------------
3. System Architecture
------------------------------------------------------------

The system consists of two main memory components:

(1) Knowledge Memory (Static)
    - The novel text
    - Split into overlapping chunks
    - Embedded into vector space
    - Stored as:
        data/chunks.json
        data/chunk_vectors.npy

(2) Experience Memory (Dynamic)
    - Stores past:
        - questions
        - answers
        - retrieved chunks
        - cited chunks
        - feedback scores
    - Enables task similarity retrieval
    - Stored as:
        data/experiences.json
        data/experience_vectors.npy

------------------------------------------------------------
4. Retrieval Method
------------------------------------------------------------

The system uses cosine similarity between embeddings.

Each chunk receives a final score:

    chunk_score =
        current_question_score
        + neighbor_weight × past_question_similarity
          × feedback_score × past_question_chunk_score
        + citation_boost

Where:

- current_question_score:
    similarity(query, chunk)

- past_question_similarity:
    similarity(current query, past query)

- past_question_chunk_score:
    similarity(past query, chunk)

- feedback_score:
    user rating of past answer (0–1)

- citation_boost:
    additional score if chunk was cited in similar past answers

This ensures that retrieval is:
- semantically relevant
- guided by similar past tasks
- improved by historical evidence

------------------------------------------------------------
5. Chunking Strategy
------------------------------------------------------------

The novel is processed as follows:

1. Split into paragraphs
2. Group paragraphs into chunks (~1200 characters)
3. Apply overlap (~200 characters) between chunks

This preserves semantic coherence and avoids boundary loss.

Each chunk contains:
- chunk_id
- text
- paragraph range

------------------------------------------------------------
6. Answer Generation
------------------------------------------------------------

The system uses an LLM to generate answers based ONLY on retrieved chunks.

Constraints:
- No external knowledge allowed
- Must cite chunk IDs (e.g., chunk_00003)
- If insufficient context → respond with uncertainty

This ensures grounded and verifiable answers.

------------------------------------------------------------
7. Learning Mechanism
------------------------------------------------------------

After each query:

1. The system stores an experience:
    - question
    - answer
    - retrieved chunks
    - cited chunks

2. User can provide feedback:
    - 1.0 = good answer
    - 0.5 = neutral
    - 0.0 = poor answer

3. Future retrieval uses:
    - task similarity (query embedding)
    - citation reuse (evidence memory)
    - feedback weighting

This allows the system to improve without retraining.

------------------------------------------------------------
8. Project Structure
------------------------------------------------------------

rag/
    config.py          # parameters
    chunking.py        # text preprocessing
    embedder.py        # embedding generation
    index_store.py     # vector storage and search
    experience.py      # memory of past queries
    retrieval.py       # task-similarity retrieval logic
    generator.py       # LLM answer generation
    rag_pipeline.py    # pipeline orchestration

Root:
    build_index.py     # builds chunk embeddings
    ask.py             # query interface
    add_feedback.py    # update feedback
    requirements.txt   # dependencies
    .env               # API key

------------------------------------------------------------
9. How to Run
------------------------------------------------------------

Step 1: Install dependencies
    python -m pip install -r requirements.txt

Step 2: Build index
    python build_index.py

Step 3: Ask questions
    python ask.py --q "Your question"

Step 4: Provide feedback (optional)
    python add_feedback.py --id <experience_id> --score 1.0

------------------------------------------------------------
10. Key Contributions
------------------------------------------------------------

- Designed a task-similarity-aware retrieval framework
- Implemented similarity-weighted multi-query fusion
- Introduced citation-based retrieval memory
- Built a feedback-aware adaptive retrieval system
- Enabled continual improvement without model retraining

------------------------------------------------------------
11. Future Work
------------------------------------------------------------

- Learn retrieval weights automatically
- Add clustering-based task routing
- Integrate large-scale vector databases (FAISS, Milvus)
- Extend to multi-document or real-world datasets

------------------------------------------------------------
12. Summary
------------------------------------------------------------

This project demonstrates how retrieval systems can be improved by incorporating 
task-level memory and experience-driven signals, providing a lightweight 
alternative to model fine-tuning for continual learning in RAG systems.

------------------------------------------------------------