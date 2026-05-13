# Retrieval Benchmark: Strategy A vs Strategy B

> Generated: 2026-05-13T06:26:20.319466+00:00  
> Corpus size: **10 documents**  
> Top-k: **3**  
> Queries evaluated: **3**

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Winner | **Strategy A (Raw)** |
| Strategy B outperforms A (query count) | 1 / 3 |
| Avg score delta (B − A) | -0.0176 |
| Avg Jaccard overlap (agreement) | 0.5667 |

---

## Query 1: `How does the system handle peak load?`

### HyDE Expansion (Strategy B input)

> During peak market hours such as market open, market close, and major economic data releases, high-frequency
trading systems experience extreme throughput demands exceeding 500,000 messages per second. Systems handle
this peak load through horizontal autoscaling of order-management microservices, kernel-bypass networking
(DPDK/RDMA), NUMA-aware CPU affinity, lock-free ring buffers, adaptive throttling when CPU utilisation exceeds
85%, and co-location within exchange data centres. Circuit breakers prevent cascading failures during flash
crashes.

### Metrics

| Metric | Strategy A | Strategy B |
|--------|-----------|-----------|
| Mean Cosine Score | 0.4440 | 0.3749 |
| Score Delta (B − A) | | **-0.0691** |
| Jaccard Overlap (A∩B / A∪B) | 1.0000 | — |
| RBO Approximation (p=0.9) | 0.2710 | — |

### Strategy A — Raw Vector Search

| Rank | Doc ID | Title | Score |
|------|--------|-------|-------|
| 1 | `doc_001` | High-Frequency Trading Systems and Peak Load Management | 0.9849 |
| 2 | `doc_005` | Stress Testing and Scenario Analysis in Financial Institutions | 0.1926 |
| 3 | `doc_004` | Market Microstructure: Liquidity, Order Books, and Price Discovery | 0.1546 |

### Strategy B — HyDE AI-Enhanced Retrieval

| Rank | Doc ID | Title | Score |
|------|--------|-------|-------|
| 1 | `doc_001` | High-Frequency Trading Systems and Peak Load Management | 0.9994 |
| 2 | `doc_005` | Stress Testing and Scenario Analysis in Financial Institutions | 0.0762 |
| 3 | `doc_004` | Market Microstructure: Liquidity, Order Books, and Price Discovery | 0.0492 |

---

## Query 2: `What are the risk management strategies for volatile markets?`

### HyDE Expansion (Strategy B input)

> Portfolio risk management employs Value-at-Risk (VaR), Conditional VaR (Expected Shortfall), and Monte Carlo
simulation to measure and limit exposure. Real-time risk limits are enforced at firm-wide, desk, and strategy
levels. Stress testing overlays historical crisis scenarios (2008 credit crisis, COVID-19 sell-off) and
reverse stress tests identify scenarios that exhaust capital buffers. GPU-accelerated engines recalculate
intraday risk across the full trading book.

### Metrics

| Metric | Strategy A | Strategy B |
|--------|-----------|-----------|
| Mean Cosine Score | 0.4765 | 0.5153 |
| Score Delta (B − A) | | **+0.0388** |
| Jaccard Overlap (A∩B / A∪B) | 0.2000 | — |
| RBO Approximation (p=0.9) | 0.0270 | — |

### Strategy A — Raw Vector Search

| Rank | Doc ID | Title | Score |
|------|--------|-------|-------|
| 1 | `doc_006` | Algorithmic Execution: TWAP, VWAP, and Smart Order Routing | 0.5344 |
| 2 | `doc_003` | Options Pricing: Black-Scholes and Volatility Surface Calibration | 0.4882 |
| 3 | `doc_008` | Credit Risk Modelling: PD, LGD, EAD, and IRB Approaches | 0.4070 |

### Strategy B — HyDE AI-Enhanced Retrieval

| Rank | Doc ID | Title | Score |
|------|--------|-------|-------|
| 1 | `doc_002` | Portfolio Risk Management and Value-at-Risk Models | 0.9735 |
| 2 | `doc_005` | Stress Testing and Scenario Analysis in Financial Institutions | 0.4148 |
| 3 | `doc_008` | Credit Risk Modelling: PD, LGD, EAD, and IRB Approaches | 0.1576 |

---

## Query 3: `How is real-time price discovery achieved in modern exchanges?`

### HyDE Expansion (Strategy B input)

> Price discovery in equity markets operates through continuous double auctions in a central limit order book
(CLOB). Market microstructure decomposes the bid-ask spread into adverse selection, inventory, and order-
processing costs. Smart order routing (SOR) algorithms route child orders across fragmented liquidity
venues—lit exchanges, dark pools, systematic internalisers—to minimise implementation shortfall. Kyle's lambda
measures market impact of large institutional orders.

### Metrics

| Metric | Strategy A | Strategy B |
|--------|-----------|-----------|
| Mean Cosine Score | 0.5426 | 0.5201 |
| Score Delta (B − A) | | **-0.0225** |
| Jaccard Overlap (A∩B / A∪B) | 0.5000 | — |
| RBO Approximation (p=0.9) | 0.2440 | — |

### Strategy A — Raw Vector Search

| Rank | Doc ID | Title | Score |
|------|--------|-------|-------|
| 1 | `doc_004` | Market Microstructure: Liquidity, Order Books, and Price Discovery | 0.8740 |
| 2 | `doc_006` | Algorithmic Execution: TWAP, VWAP, and Smart Order Routing | 0.4338 |
| 3 | `doc_007` | Real-Time Settlement Systems and Post-Trade Infrastructure | 0.3199 |

### Strategy B — HyDE AI-Enhanced Retrieval

| Rank | Doc ID | Title | Score |
|------|--------|-------|-------|
| 1 | `doc_004` | Market Microstructure: Liquidity, Order Books, and Price Discovery | 0.9747 |
| 2 | `doc_006` | Algorithmic Execution: TWAP, VWAP, and Smart Order Routing | 0.4083 |
| 3 | `doc_005` | Stress Testing and Scenario Analysis in Financial Institutions | 0.1773 |

---

## Design Notes

### Similarity Metric: Why Cosine over Euclidean?

| Criterion | Cosine Similarity | Euclidean (L2) Distance |
|-----------|------------------|------------------------|
| Magnitude invariance | ✅ (chunk length doesn't matter) | ❌ Penalises longer passages |
| Training alignment | ✅ sentence-transformers trained with cosine loss | ❌ Geometric mismatch |
| FAISS index | `IndexFlatIP` on L2-normalised vectors | `IndexFlatL2` |
| Finance corpora | ✅ Short queries vs long paragraphs → cosine wins | ❌ |

### Vertex AI Vector Search (Matching Engine) Migration

1. **Create index** via `POST projects/{project}/locations/{region}/indexes`  
   Set `distanceMeasureType: COSINE_DISTANCE` and `algorithmConfig: treeAhConfig` (ScaNN ANN).  
2. **Upsert datapoints** using `IndexService.UpsertDatapoints` with float32 vectors + string IDs.  
3. **Deploy** to an `IndexEndpoint` (Online Matching) or batch-match via `BatchMatchService`.  
4. **Query** with `MatchService.FindNeighbors(deployed_index_id, queries, neighbor_count)`.  
5. **Metadata filtering**: add numeric/string *restricts* on doc metadata fields for hybrid filters.  
6. **Scaling**: Matching Engine shards the index across GCP infrastructure with auto-replication;  
   swap `FAISSVectorStore.search()` for `MatchService` calls with no other pipeline changes.  
