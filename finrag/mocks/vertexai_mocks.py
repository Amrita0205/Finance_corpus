"""
finrag/mocks/vertexai_mocks.py
==============================
Drop-in mocks for Google Cloud Vertex AI SDK objects used in production:
  - vertexai.language_models.TextEmbeddingModel
  - vertexai.generative_models.GenerativeModel

These mocks are used both in tests (via pytest monkeypatch / unittest.mock)
and at runtime when the real SDK is unavailable, so the pipeline can run
entirely offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Vertex AI TextEmbeddingModel mock
# ---------------------------------------------------------------------------

@dataclass
class MockTextEmbeddingInstance:
    """Mimics vertexai.language_models.TextEmbeddingInput."""
    text: str
    task_type: str = "RETRIEVAL_DOCUMENT"


@dataclass
class MockTextEmbedding:
    """Mimics the object returned by TextEmbeddingModel.get_embeddings()."""
    values: list[float]


class MockTextEmbeddingModel:
    """
    Mock of ``vertexai.language_models.TextEmbeddingModel``.

    In production this delegates to Vertex AI's ``textembedding-gecko`` model.
    Here it delegates to a locally-loaded ``sentence-transformers`` encoder
    so that tests are deterministic and reproducible without any GCP credentials.

    Usage
    -----
    >>> model = MockTextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    >>> [emb] = model.get_embeddings(["What is VaR?"])
    >>> len(emb.values)
    384
    """

    # Lazily initialised so import is fast even when the library is absent.
    _encoder: Any = field(default=None, init=False)
    _model_name: str = "local-tfidf-svd-384"

    # Shared corpus-fitted encoder (fitted once for all instances).
    _shared_encoder: Any = None

    def __init__(self, model_name: str = "textembedding-gecko@003") -> None:
        self._vertex_model_name = model_name
        self._encoder = None  # lazy per-instance reference

    @classmethod
    def from_pretrained(cls, model_name: str) -> "MockTextEmbeddingModel":
        """Mirror Vertex AI's factory method signature."""
        return cls(model_name=model_name)

    def _get_encoder(self):
        """
        Returns a fitted TF-IDF + TruncatedSVD pipeline that produces
        384-dimensional dense embeddings entirely from local libraries
        (scikit-learn + numpy) without any network access.

        The encoder is lazily fitted on first call using the finance corpus
        so embeddings are domain-aware out of the box.
        """
        if MockTextEmbeddingModel._shared_encoder is not None:
            return MockTextEmbeddingModel._shared_encoder

        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.decomposition import TruncatedSVD  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore

        # Import the finance corpus for fitting
        try:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
            from data.finance_corpus import FINANCE_CORPUS
            fit_texts = [d["content"] for d in FINANCE_CORPUS]
        except ImportError:
            # Fallback: generic finance seed texts if corpus not yet importable
            fit_texts = [
                "high frequency trading peak load market orders",
                "portfolio risk management value at risk VaR",
                "options pricing Black-Scholes volatility surface",
                "market microstructure liquidity order book",
                "stress testing Basel III regulatory capital",
                "algorithmic execution TWAP VWAP smart order routing",
                "settlement clearing counterparty central securities",
                "credit risk probability of default loss given default",
                "quantitative factor model Fama-French momentum",
                "real-time price discovery bid ask spread",
            ]

        # Produce 384-dim dense embeddings via LSA (TF-IDF + SVD)
        # n_components capped to min(vocabulary-1, 384) to avoid SVD errors
        n_comp = min(384, len(fit_texts) - 1, 300)
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
                strip_accents="unicode",
            )),
            ("svd", TruncatedSVD(n_components=n_comp, random_state=42)),
        ])
        pipe.fit(fit_texts)
        MockTextEmbeddingModel._shared_encoder = pipe
        return pipe

    def get_embeddings(
        self,
        texts: list[str] | list[MockTextEmbeddingInstance],
    ) -> list[MockTextEmbedding]:
        """Return a list of MockTextEmbedding, one per input."""
        raw_texts = [
            t.text if isinstance(t, MockTextEmbeddingInstance) else t
            for t in texts
        ]
        encoder = self._get_encoder()
        # TfidfVectorizer + SVD pipeline: transform returns (n, n_components)
        import numpy as np
        matrix = encoder.transform(raw_texts).astype(np.float32)
        # Pad to exactly 384 dims if SVD produced fewer components
        if matrix.shape[1] < 384:
            pad = np.zeros((matrix.shape[0], 384 - matrix.shape[1]), dtype=np.float32)
            matrix = np.hstack([matrix, pad])
        return [MockTextEmbedding(values=vec.tolist()) for vec in matrix]


# ---------------------------------------------------------------------------
# Finance-domain HyDE query expander (mock GenerativeModel)
# ---------------------------------------------------------------------------

# Maps key financial trigger phrases → richer hypothetical-document expansions.
# A real production system would call PaLM 2 / Gemini via Vertex AI.
_FINANCE_EXPANSIONS: dict[str, str] = {
    "peak load": (
        "During peak market hours such as market open, market close, and major economic data releases, "
        "high-frequency trading systems experience extreme throughput demands exceeding 500,000 messages "
        "per second. Systems handle this peak load through horizontal autoscaling of order-management "
        "microservices, kernel-bypass networking (DPDK/RDMA), NUMA-aware CPU affinity, lock-free ring "
        "buffers, adaptive throttling when CPU utilisation exceeds 85%, and co-location within exchange "
        "data centres. Circuit breakers prevent cascading failures during flash crashes."
    ),
    "risk management": (
        "Portfolio risk management employs Value-at-Risk (VaR), Conditional VaR (Expected Shortfall), "
        "and Monte Carlo simulation to measure and limit exposure. Real-time risk limits are enforced at "
        "firm-wide, desk, and strategy levels. Stress testing overlays historical crisis scenarios (2008 "
        "credit crisis, COVID-19 sell-off) and reverse stress tests identify scenarios that exhaust capital "
        "buffers. GPU-accelerated engines recalculate intraday risk across the full trading book."
    ),
    "price discovery": (
        "Price discovery in equity markets operates through continuous double auctions in a central limit "
        "order book (CLOB). Market microstructure decomposes the bid-ask spread into adverse selection, "
        "inventory, and order-processing costs. Smart order routing (SOR) algorithms route child orders "
        "across fragmented liquidity venues—lit exchanges, dark pools, systematic internalisers—to minimise "
        "implementation shortfall. Kyle's lambda measures market impact of large institutional orders."
    ),
    "settlement": (
        "Post-trade settlement involves CCPs, CSDs, and custodian banks. Central counterparties novate "
        "trades, eliminating bilateral counterparty risk via multilateral netting. The shift from T+2 to "
        "T+1 settlement demands real-time matching. TARGET2-Securities uses DvP to ensure simultaneous "
        "exchange of cash and securities. RTGS systems manage intraday liquidity queues."
    ),
    "credit risk": (
        "Credit risk under the IRB approach requires estimating Probability of Default (PD), Loss Given "
        "Default (LGD), and Exposure at Default (EAD). CVA measures counterparty default risk on OTC "
        "derivatives. Wrong-Way Risk arises when counterparty quality correlates negatively with derivative "
        "value. Downturn LGD applies stressed recovery rates."
    ),
    "capital": (
        "Basel III/IV mandates minimum CET1 ratios, leverage ratio floors, and output floors. FRTB replaces "
        "VaR with Expected Shortfall for market risk capital. NSFR and LCR ensure stable funding and liquid "
        "asset buffers. Pillar 2 SREP adds institution-specific capital requirements above the Pillar 1 minimum."
    ),
    "volatility": (
        "Implied volatility surfaces are calibrated from option chain prices using local volatility (Dupire) "
        "or stochastic volatility models (Heston, SABR). The volatility smile reflects fat tails and skewness "
        "not captured by Black-Scholes. Real-time surface calibration pipelines process streaming quotes within "
        "50 ms. Greeks (delta, vega, gamma) drive hedging and risk decomposition."
    ),
    "execution": (
        "Algorithmic execution uses TWAP, VWAP, and implementation shortfall strategies to minimise market "
        "impact when liquidating large positions. Adaptive algorithms monitor real-time fill rates, book "
        "imbalance, and spread widening to pace execution dynamically. SOR routes slices across venues for "
        "best execution using the Almgren-Chriss optimal execution model."
    ),
    "factor": (
        "Multi-factor equity models—Fama-French 5-factor, momentum (WML)—decompose returns into systematic "
        "risk premia. Alternative data (satellite imagery, credit-card panels, NLP sentiment from earnings "
        "transcripts) augments traditional factors. Ledoit-Wolf shrinkage produces well-conditioned "
        "covariance matrices for mean-variance optimisation across large universes."
    ),
    "retail": (
        "Retail brokerage platforms route customer orders to wholesale market makers via Payment for Order "
        "Flow (PFOF), internalising uninformed retail flow at a small price improvement over the NBBO. "
        "During extreme events like the GameStop short squeeze, clearing brokers imposed intraday margin "
        "requirements forcing platforms to restrict buy-side order flow. Engineering these platforms "
        "requires sub-100 ms acknowledgement latency, real-time margin calculation across millions of "
        "accounts, in-memory data grids (Redis, Hazelcast), and event-sourced ledger architectures."
    ),
    "short squeeze": (
        "A short squeeze occurs when heavily shorted stocks rise sharply, forcing short sellers to cover "
        "positions, amplifying price moves. Retail platforms like Robinhood faced clearing margin calls "
        "during the 2021 GameStop event, exposing PFOF infrastructure dependencies. Market makers "
        "internalise retail order flow, and regulators debate whether PFOF conflicts with best execution."
    ),
    "order routing": (
        "Smart order routing distributes retail and institutional orders across venues—lit exchanges, "
        "dark pools, and wholesale market makers—based on price, fee, and fill probability. Retail "
        "brokers using PFOF route most flow to internalising market makers who offer price improvement "
        "over NBBO. Order routing decisions must balance execution quality, latency, and regulatory "
        "best-execution obligations under Reg NMS."
    ),
}


@dataclass
class MockGenerativeResponse:
    """Mimics vertexai.generative_models.GenerationResponse."""
    text: str


class MockGenerativeModel:
    """
    Mock of ``vertexai.generative_models.GenerativeModel`` (Gemini / PaLM 2).

    Implements **HyDE** (Hypothetical Document Embeddings): given a user query,
    it generates a short hypothetical answer document. Embedding this richer
    passage instead of the terse query significantly improves recall for
    semantic search — especially on domain-specific corpora.

    In production the ``generate_content`` call routes to Vertex AI Gemini.
    Here it uses a keyword-match expansion table for deterministic offline use.

    Usage
    -----
    >>> model = MockGenerativeModel("gemini-1.0-pro")
    >>> resp = model.generate_content("How does the system handle peak load?")
    >>> print(resp.text[:60])
    During peak market hours such as market open, market close
    """

    SYSTEM_PROMPT = (
        "You are a senior quantitative finance engineer. "
        "Given a user question, write a concise technical paragraph (≤120 words) "
        "that a relevant document in our corpus would contain. "
        "Use precise financial and engineering terminology. "
        "Do NOT answer the question directly; instead write as if you are the document."
    )

    def __init__(self, model_name: str = "gemini-1.0-pro") -> None:
        self._vertex_model_name = model_name

    @classmethod
    def from_pretrained(cls, model_name: str) -> "MockGenerativeModel":
        return cls(model_name=model_name)

    def generate_content(self, prompt: str) -> MockGenerativeResponse:
        """
        Expand the user query into a hypothetical finance document paragraph.

        Matches against the internal expansion table; falls back to a generic
        enrichment that appends domain synonyms if no key phrase is found.
        """
        lower = prompt.lower()

        for trigger, expansion in _FINANCE_EXPANSIONS.items():
            if trigger in lower:
                return MockGenerativeResponse(text=expansion)

        # Generic fallback: echo the query with financial terminology enrichment
        enriched = (
            f"{prompt.strip()} "
            "This is relevant to quantitative finance, risk modelling, algorithmic trading, "
            "portfolio management, and regulatory capital requirements in modern financial markets."
        )
        return MockGenerativeResponse(text=enriched)

    # ------------------------------------------------------------------
    # Vertex AI SDK compatibility shims
    # ------------------------------------------------------------------

    def start_chat(self, **_kwargs):  # noqa: ANN001
        """Stub for multi-turn chat sessions (not used here)."""
        return self

    def send_message(self, message: str, **_kwargs) -> MockGenerativeResponse:
        return self.generate_content(message)


# ---------------------------------------------------------------------------
# Module-level factory helpers (mirror vertexai SDK top-level functions)
# ---------------------------------------------------------------------------

def init(project: str = "mock-project", location: str = "us-central1") -> None:  # noqa: ARG001
    """No-op mock of ``vertexai.init()``."""


def mock_vertexai_module() -> dict:
    """
    Return a dict suitable for use with ``unittest.mock.patch.dict(sys.modules, ...)``
    to inject the mock SDK before any import of ``vertexai``.
    """
    import types

    vertexai_mod = types.ModuleType("vertexai")
    vertexai_mod.init = init  # type: ignore[attr-defined]

    language_models_mod = types.ModuleType("vertexai.language_models")
    language_models_mod.TextEmbeddingModel = MockTextEmbeddingModel  # type: ignore[attr-defined]
    language_models_mod.TextEmbeddingInput = MockTextEmbeddingInstance  # type: ignore[attr-defined]

    generative_models_mod = types.ModuleType("vertexai.generative_models")
    generative_models_mod.GenerativeModel = MockGenerativeModel  # type: ignore[attr-defined]

    return {
        "vertexai": vertexai_mod,
        "vertexai.language_models": language_models_mod,
        "vertexai.generative_models": generative_models_mod,
    }