"""
run_evaluation.py
Runs RAGAS evaluation against the live pipeline.
Execute this after any significant change to retrieval or generation.
Results go into README as pipeline quality evidence.
"""
import asyncio
import logging
from query import QueryEngine
from generation.generator import Generator
from evaluation.ragas_evaluator import RAGASEvaluator
from evaluation.test_dataset import TEST_DATASET

logging.basicConfig(level=logging.WARNING)  # suppress INFO noise during eval


async def collect_pipeline_outputs() -> tuple:
    """Run each test question through the full pipeline."""
    engine    = QueryEngine()
    generator = Generator()

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in TEST_DATASET:
        print(f"  Evaluating: {item['question'][:60]}...")

        chunks = await engine.run(item["question"])
        result = await generator.generate(item["question"], chunks)

        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append([c.text for c in chunks])
        ground_truths.append(item["ground_truth"])

    return questions, answers, contexts, ground_truths


async def main():
    print("\nCollecting pipeline outputs...")
    questions, answers, contexts, ground_truths = \
        await collect_pipeline_outputs()

    print("\nRunning RAGAS evaluation (this takes 2-3 minutes)...")
    evaluator = RAGASEvaluator()
    scores = evaluator.evaluate(questions, answers, contexts, ground_truths)

    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    for metric, score in scores.items():
        bar    = "█" * int(score * 20)
        status = "✓" if score >= 0.7 else "⚠"
        print(f"{status} {metric:<22} {score:.4f}  {bar}")
    print("=" * 50)
    print("\nTarget: all metrics above 0.70 for production readiness")

    # Add at the end of main() after printing scores:
    THRESHOLDS = {
        "faithfulness": 0.85,
        "context_precision": 0.75,
    }

    failed = [
        f"{m}={s:.4f} < {THRESHOLDS[m]}"
        for m, s in scores.items()
        if m in THRESHOLDS and s < THRESHOLDS[m]
    ]

    if failed:
        print(f"\n❌ EVALUATION FAILED — metrics below threshold: {failed}")
        print("Deployment blocked. Fix retrieval pipeline before merging.")
        import sys
        sys.exit(1)
    else:
        print("\n✅ All critical metrics within thresholds — safe to deploy.")
asyncio.run(main())
