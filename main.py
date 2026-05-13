"""
main.py
=======
Entry point for the Finance RAG Benchmark.

Run with:
    python main.py

Outputs:
    retrieval_benchmark.md   — Markdown comparison report
    retrieval_benchmark.json — Machine-readable comparison report
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Make sure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("finrag.main")


def main() -> None:
    from data.finance_corpus import FINANCE_CORPUS
    from finrag.mocks.vertexai_mocks import MockGenerativeModel
    from finrag.pipeline.rag_pipeline import RAGPipeline
    from finrag.benchmarking.benchmark import Benchmarker

    logger.info("=" * 60)
    logger.info("Finance RAG Benchmark  —  Context-Aware Retrieval Engine")
    logger.info("=" * 60)

    # 1. Build and ingest pipeline
    logger.info("Initialising pipeline …")
    pipeline = RAGPipeline(
        generative_model=MockGenerativeModel("gemini-1.0-pro"),
        use_hybrid_hyde=False,
    )
    pipeline.ingest(FINANCE_CORPUS)
    logger.info("Pipeline ready: %s", pipeline)

    # 2. Quick smoke test — show expansions for the three benchmark queries
    benchmark_queries = [
        "How does the system handle peak load?",
        "What are the risk management strategies for volatile markets?",
        "How is real-time price discovery achieved in modern exchanges?",
        "How do retail brokers handle order routing and what happens during a short squeeze?",
    ]

    print("\n" + "=" * 60)
    print("HyDE Query Expansions (Strategy B)")
    print("=" * 60)
    for q in benchmark_queries:
        expansion = pipeline.get_hyde_expansion(q)
        print(f"\nQuery   : {q}")
        print(f"Expanded: {expansion[:200]}…")

    # 3. Run benchmark
    print("\n" + "=" * 60)
    print("Running benchmark …")
    print("=" * 60)
    benchmarker = Benchmarker(pipeline, queries=benchmark_queries, top_k=3)
    report = benchmarker.run()

    # 4. Print summary
    print("\n── Summary ──────────────────────────────────────────────────")
    print(json.dumps(report.summary, indent=2))

    # 5. Print per-query comparison table
    print("\n── Per-Query Results ────────────────────────────────────────")
    for m in report.results:
        print(f"\n  Query: {m.query}")
        print(f"  Strategy A mean score : {m.strategy_a_mean_score:.4f}")
        print(f"  Strategy B mean score : {m.strategy_b_mean_score:.4f}")
        print(f"  Score delta (B − A)   : {m.score_delta:+.4f}")
        print(f"  Jaccard overlap       : {m.jaccard_overlap:.4f}")
        print(f"  RBO approx (p=0.9)    : {m.rbo_approx:.4f}")
        print()
        print("  Strategy A top-3:")
        for r in m.strategy_a_results:
            print(f"    #{r['rank']}  {r['doc_id']}  {r['score']:.4f}  {r['title']}")
        print("  Strategy B top-3:")
        for r in m.strategy_b_results:
            print(f"    #{r['rank']}  {r['doc_id']}  {r['score']:.4f}  {r['title']}")

    # 6. Write reports
    repo_root = Path(__file__).parent
    md_path = repo_root / "retrieval_benchmark.md"
    json_path = repo_root / "retrieval_benchmark.json"

    benchmarker.to_markdown(report, path=str(md_path))
    benchmarker.to_json(report, path=str(json_path))

    print("\n" + "=" * 60)
    print(f"✅  Reports written:")
    print(f"     {md_path}")
    print(f"     {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()