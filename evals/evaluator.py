"""
Evaluation Framework - assess agent performance on known Q&A pairs
"""

import json
import os
from typing import List, Dict, Tuple, Any

from fastapi.testclient import TestClient
from sentence_transformers import SentenceTransformer, util

from api.main import app


class AgentEvaluator:
    def __init__(self, golden_qa_path: str = "data/golden_qa.json"):
        self.golden_qa_path = golden_qa_path
        self.golden_qa = self._load_golden_qa()
        self.client = TestClient(app)

        # Load evaluation model once
        self.eval_model = SentenceTransformer("all-MiniLM-L6-v2")

    def _load_golden_qa(self) -> List[Dict[str, str]]:
        """Load golden Q&A pairs for evaluation"""
        if not os.path.exists(self.golden_qa_path):
            return []

        with open(self.golden_qa_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_single(
        self,
        question: str,
        expected_answer: str
    ) -> Dict[str, Any]:
        """Evaluate a single question"""

        response = self.client.post(
            "/ask",
            json={"question": question}
        )

        if response.status_code != 200:
            return {
                "question": question,
                "expected_answer": expected_answer,
                "passed": False,
                "error": "API returned error",
                "status_code": response.status_code
            }

        data = response.json()

        answer = data.get("answer", "")
        sources = data.get("sources", [])
        confidence = data.get("confidence", "")

        # Simple term-overlap metric
        expected_terms = set(expected_answer.lower().split()[:5])
        answer_terms = set(answer.lower().split())

        term_overlap = (
            len(expected_terms & answer_terms) / len(expected_terms)
            if expected_terms
            else 0
        )

        # Semantic similarity metric
        expected_emb = self.eval_model.encode(
            expected_answer,
            convert_to_tensor=True
        )

        actual_emb = self.eval_model.encode(
            answer,
            convert_to_tensor=True
        )

        similarity = float(
            util.cos_sim(expected_emb, actual_emb).item()
        )

        passed = similarity >= 0.6

        return {
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": answer,
            "sources": sources,
            "confidence": confidence,
            "term_overlap": round(term_overlap, 3),
            "semantic_similarity": round(similarity, 3),
            "passed": passed
        }

    def evaluate_all(self) -> Tuple[float, List[Dict[str, Any]]]:
        """Evaluate all golden Q&A pairs"""

        results = []

        for qa_pair in self.golden_qa:
            result = self.evaluate_single(
                qa_pair["question"],
                qa_pair["answer"]
            )
            results.append(result)

        passed_count = sum(
            1 for r in results if r.get("passed", False)
        )

        total = len(results)

        accuracy = (
            passed_count / total * 100
            if total > 0
            else 0
        )

        return accuracy, results

    def print_report(
        self,
        accuracy: float,
        results: List[Dict[str, Any]]
    ) -> None:
        """Print evaluation report"""

        print("\n" + "=" * 80)
        print("EVALUATION REPORT")
        print("=" * 80)

        passed_count = sum(
            1 for r in results if r.get("passed", False)
        )

        print(
            f"Accuracy: {accuracy:.1f}% "
            f"({passed_count}/{len(results)})\n"
        )

        for i, result in enumerate(results, 1):
            status = "✓ PASS" if result.get("passed") else "✗ FAIL"

            print(
                f"{i}. {status} | "
                f"{result['question'][:50]}..."
            )

            print(
                f"   Similarity: "
                f"{result.get('semantic_similarity', 0):.3f}"
            )

            if not result.get("passed"):
                print(
                    f"   Expected: "
                    f"{result.get('expected_answer', '')[:60]}..."
                )
                print(
                    f"   Got: "
                    f"{result.get('actual_answer', '')[:60]}..."
                )

            print()

        print("=" * 80)


if __name__ == "__main__":
    evaluator = AgentEvaluator()

    if evaluator.golden_qa:
        accuracy, results = evaluator.evaluate_all()
        evaluator.print_report(accuracy, results)
    else:
        print(
            "No golden Q&A pairs found. "
            "Create data/golden_qa.json first."
        )