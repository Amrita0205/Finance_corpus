"""
tests/test_benchmark.py
========================
Tests for the Benchmarker and output rendering.
"""

from __future__ import annotations

import json

import pytest

from finrag.benchmarking.benchmark import Benchmarker, BenchmarkReport, QueryMetrics


class TestBenchmarker:

    @pytest.fixture(scope="class")
    def benchmark_report(self, ingested_pipeline):
        benchmarker = Benchmarker(ingested_pipeline, top_k=3)
        return benchmarker.run()

    def test_report_is_benchmark_report(self, benchmark_report):
        assert isinstance(benchmark_report, BenchmarkReport)

    def test_report_has_correct_query_count(self, benchmark_report):
        assert benchmark_report.queries_evaluated == 4

    def test_report_results_length(self, benchmark_report):
        assert len(benchmark_report.results) == 4

    def test_each_result_is_query_metrics(self, benchmark_report):
        assert all(isinstance(m, QueryMetrics) for m in benchmark_report.results)

    def test_strategy_a_results_populated(self, benchmark_report):
        for m in benchmark_report.results:
            assert len(m.strategy_a_results) == 3

    def test_strategy_b_results_populated(self, benchmark_report):
        for m in benchmark_report.results:
            assert len(m.strategy_b_results) == 3

    def test_mean_scores_in_range(self, benchmark_report):
        for m in benchmark_report.results:
            assert 0.0 <= m.strategy_a_mean_score <= 1.0
            assert 0.0 <= m.strategy_b_mean_score <= 1.0

    def test_jaccard_in_range(self, benchmark_report):
        for m in benchmark_report.results:
            assert 0.0 <= m.jaccard_overlap <= 1.0

    def test_rbo_in_range(self, benchmark_report):
        for m in benchmark_report.results:
            assert 0.0 <= m.rbo_approx <= 1.0

    def test_score_delta_computed(self, benchmark_report):
        for m in benchmark_report.results:
            expected = round(m.strategy_b_mean_score - m.strategy_a_mean_score, 4)
            assert abs(m.score_delta - expected) < 1e-4

    def test_hyde_expansion_non_empty(self, benchmark_report):
        for m in benchmark_report.results:
            assert isinstance(m.hyde_expansion, str)
            assert len(m.hyde_expansion) > 10

    def test_summary_winner_field(self, benchmark_report):
        assert benchmark_report.summary["winner"] in {
            "Strategy A (Raw)", "Strategy B (HyDE)", "Tie"
        }

    def test_corpus_size_in_report(self, benchmark_report, full_corpus):
        assert benchmark_report.corpus_size == len(full_corpus)


class TestBenchmarkerOutput:

    @pytest.fixture(scope="class")
    def benchmarker_and_report(self, ingested_pipeline):
        b = Benchmarker(ingested_pipeline, top_k=3)
        report = b.run()
        return b, report

    def test_to_json_is_valid_json(self, benchmarker_and_report):
        b, report = benchmarker_and_report
        json_str = b.to_json(report)
        data = json.loads(json_str)
        assert "results" in data

    def test_to_json_saves_file(self, benchmarker_and_report, tmp_path):
        b, report = benchmarker_and_report
        out = str(tmp_path / "bench.json")
        b.to_json(report, path=out)
        import pathlib
        assert pathlib.Path(out).exists()
        content = pathlib.Path(out).read_text()
        assert len(content) > 0

    def test_to_markdown_contains_headers(self, benchmarker_and_report):
        b, report = benchmarker_and_report
        md = b.to_markdown(report)
        assert "# Retrieval Benchmark" in md
        assert "Strategy A" in md
        assert "Strategy B" in md

    def test_to_markdown_saves_file(self, benchmarker_and_report, tmp_path):
        b, report = benchmarker_and_report
        out = str(tmp_path / "bench.md")
        b.to_markdown(report, path=out)
        import pathlib
        content = pathlib.Path(out).read_text(encoding="utf-8")
        assert "# Retrieval Benchmark" in content

    def test_custom_queries(self, ingested_pipeline):
        custom = ["What is VaR?", "Explain VWAP execution"]
        b = Benchmarker(ingested_pipeline, queries=custom, top_k=2)
        report = b.run()
        assert report.queries_evaluated == 2
        assert [m.query for m in report.results] == custom