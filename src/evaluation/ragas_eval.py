# src/evaluation/ragas_eval.py
import os
import uuid
import logging
from datetime import datetime
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from sqlalchemy import create_engine, text
from .golden_dataset import GOLDEN_DATASET
from ..generation.generator import Generator

logger = logging.getLogger(__name__)

class RAGASEvaluator:
    def __init__(self):
        self.generator = Generator()
        self.engine = create_engine(os.environ["DATABASE_URL"])

    def run(self) -> dict:
        run_id = str(uuid.uuid4())[:8]
        logger.info(f"Starting RAGAS eval run {run_id} "
                    f"on {len(GOLDEN_DATASET)} questions")
        questions, answers, contexts, ground_truths = [], [], [], []

        for item in GOLDEN_DATASET:
            result = self.generator.answer(item["question"])
            questions.append(item["question"])
            answers.append(result["answer"])
            contexts.append([d["context"] for d in result["context"]])
            ground_truths.append(item["ground_truth"])

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        scores = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )

        result = {
            "run_id": run_id,
            "faithfulness": float(scores["faithfulness"]),
            "answer_relevancy": float(scores["answer_relevancy"]),
            "context_precision": float(scores["context_precision"]),
            "num_questions": len(GOLDEN_DATASET),
        }

        self._store_result(result)
        logger.info(f"RUN {run_id} complete: {result}")
        return result

    def _store_result(self, result: dict):
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO eval_results
                        (run_id, faithfulness, answer_relevancy,
                        context_precision, context_recall, num_questions)
                    VALUES
                        (:run_id, :faithfulness, :answer_relevancy,
                         :context_precision, :context_recall, :num_questions)
                """),
                result,
            )