# src/evaluation/golden_dataset.py
"""
Manually curate 30-50 Q&A pairs. Be honest - tedious is the point.
Include 3 types:
    - retrieval-only: answer is directly in a document
    - multi-hop: answer requires combining 2+ documents
    - negative: answer is NOT in the corpus (test hallucination)
"""

GOLDEN_DATASET = [
    {
        "question": "What chunking strategy gives better recall for multi-sentence answers?",
        "ground_truth": "Semantic chunking preserves sentence boundaries and tends to give better recall for answers that span multiple sentences, compared to fixed-size recursive splitting.",
        "source_docs": ["chunking_strategies.txt"],
        "category": "retrieval-only",
    },
    {
        "question": "How does RRF fusion combine dense and sparse retrieval scores?",
        "ground_truth": "Reciprocal Rank Fusion assigns each document a score of 1/(k + rank) for each retrieval method, then sums these scores across methods. k=60 is standard.",
        "source_docs": ["hybrid_search.txt"],
        "category": "retrieval-only",
    },
    {
        "question": "What is the cost of running 1000 embedding calls with text-embedding-3-small?",
        "ground_truth": "I don't have enough information to answer that.",
        "source_docs": [],
        "category": "negative", # not in corpus - should NOT hallucinate
    },
    # Add 27+ more...
]