"""
Finance Domain Corpus
=====================
10 technical paragraphs covering core areas of modern quantitative finance,
trading infrastructure, and risk management. Chosen to be semantically rich
and overlap with common recruiter/assessor benchmark queries.
"""

FINANCE_CORPUS: list[dict] = [
    {
        "id": "doc_001",
        "title": "High-Frequency Trading Systems and Peak Load Management",
        "content": (
            "High-frequency trading (HFT) systems are purpose-built to sustain extreme throughput during "
            "peak market events such as market open, economic data releases, and earnings announcements. "
            "Co-location of trading servers within exchange data centres reduces round-trip latency to "
            "sub-microsecond levels, achieved via kernel-bypass networking (DPDK/RDMA) and lock-free "
            "ring-buffer queues. During peak load, these platforms sustain order submission rates exceeding "
            "500,000 messages per second by distributing workloads across NUMA-aware CPU cores with strict "
            "CPU affinity. Adaptive throttling mechanisms activate when system CPU utilisation breaches 85%, "
            "shedding lower-priority order flows to protect critical market-making pipelines. Horizontal "
            "autoscaling of order-management microservices allows the system to elastically absorb 10× "
            "normal volume spikes without degrading median latency beyond 2 µs."
        ),
    },
    {
        "id": "doc_002",
        "title": "Portfolio Risk Management and Value-at-Risk Models",
        "content": (
            "Modern portfolio risk frameworks combine parametric Value-at-Risk (VaR) with Monte Carlo "
            "simulation to capture fat-tailed loss distributions that Gaussian models underestimate. "
            "Conditional VaR (CVaR), also known as Expected Shortfall, is favoured under Basel IV "
            "because it measures the mean loss beyond the VaR threshold, providing a coherent risk "
            "measure. Risk limits are decomposed hierarchically: firm-wide VaR → desk-level VaR → "
            "strategy-level greeks. Real-time intraday risk recalculation uses GPU-accelerated Monte "
            "Carlo engines (CUDA) capable of pricing 100,000 paths per second. Stress-testing overlays "
            "predefined shock scenarios (2008 credit crisis, 2020 COVID sell-off) onto current positions, "
            "while reverse stress tests identify the scenario that would exactly exhaust the capital buffer."
        ),
    },
    {
        "id": "doc_003",
        "title": "Options Pricing: Black-Scholes and Volatility Surface Calibration",
        "content": (
            "The Black-Scholes-Merton (BSM) framework prices European options under assumptions of "
            "log-normal asset returns, constant volatility, and frictionless markets. In practice, "
            "traders observe a volatility smile / skew surface — implied volatilities differ across "
            "strikes and tenors — which BSM cannot explain. Local volatility models (Dupire) calibrate "
            "a deterministic σ(S,t) surface to fit market prices exactly, while stochastic volatility "
            "models (Heston, SABR) capture mean-reverting variance dynamics. The Greeks (delta, gamma, "
            "vega, theta, rho) quantify first- and second-order sensitivities of option prices to market "
            "inputs and are used for hedging and risk decomposition. Real-time vol-surface calibration "
            "pipelines ingest streaming option chain quotes, solve the inverse problem via Tikhonov "
            "regularisation, and publish a consistent surface to downstream pricing libraries within 50 ms."
        ),
    },
    {
        "id": "doc_004",
        "title": "Market Microstructure: Liquidity, Order Books, and Price Discovery",
        "content": (
            "Price discovery in modern equity markets occurs through a continuous double auction mechanism "
            "where a central limit order book (CLOB) aggregates limit orders from thousands of participants. "
            "Market microstructure theory decomposes the bid-ask spread into three components: inventory "
            "cost, adverse selection cost, and order-processing cost. Kyle's lambda (market impact) "
            "measures how aggressively large orders move prices, informing optimal execution algorithms. "
            "Tick-by-tick Level-2 data, capturing full order-book depth, is stored in columnar formats "
            "(Apache Parquet, KDB+) for latency-sensitive backtesting. Fragmented liquidity across dark "
            "pools, lit exchanges, and systematic internalisers (SIs) requires smart order routing (SOR) "
            "algorithms to minimise implementation shortfall while navigating venue-specific fee schedules."
        ),
    },
    {
        "id": "doc_005",
        "title": "Stress Testing and Scenario Analysis in Financial Institutions",
        "content": (
            "Regulatory stress testing (DFAST, EBA stress tests) requires financial institutions to model "
            "their balance sheets under adverse macroeconomic scenarios defined by the regulator. Internal "
            "stress frameworks augment regulatory scenarios with firm-specific idiosyncratic shocks such as "
            "a key counterparty default or a sudden withdrawal of repo funding. Scenario analysis engines "
            "propagate shocks through a dependency graph of risk factors, re-pricing the entire trading book "
            "using full revaluation rather than sensitivity approximations. Liquidity stress tests measure "
            "the Net Stable Funding Ratio (NSFR) and Liquidity Coverage Ratio (LCR) under a 30-day acute "
            "stress horizon. Results feed directly into the Internal Capital Adequacy Assessment Process "
            "(ICAAP), determining how much CET1 capital must be held as a management buffer above the "
            "regulatory minimum."
        ),
    },
    {
        "id": "doc_006",
        "title": "Algorithmic Execution: TWAP, VWAP, and Smart Order Routing",
        "content": (
            "Algorithmic execution strategies minimise market impact by intelligently slicing large parent "
            "orders into smaller child orders dispatched over time. Time-Weighted Average Price (TWAP) "
            "algorithms distribute execution uniformly, while Volume-Weighted Average Price (VWAP) "
            "algorithms schedule child orders proportional to the historical intraday volume profile to "
            "track the benchmark price. Arrival-price (implementation shortfall) strategies optimise the "
            "trade-off between timing risk and market impact using the Almgren-Chriss model. Adaptive "
            "algorithms monitor real-time fill rates, spread widening, and order-book imbalance signals, "
            "pausing or accelerating execution dynamically. Smart Order Routing (SOR) evaluates quote "
            "quality across multiple venues simultaneously, routing each slice to the venue offering the "
            "best combination of price, fee, and fill probability."
        ),
    },
    {
        "id": "doc_007",
        "title": "Real-Time Settlement Systems and Post-Trade Infrastructure",
        "content": (
            "Post-trade infrastructure processes trade confirmations, clearing, and settlement through a "
            "chain of Central Counterparties (CCPs), Central Securities Depositories (CSDs), and custodian "
            "banks. CCPs interpose themselves between buyer and seller via novation, becoming the buyer to "
            "every seller and the seller to every buyer, thereby eliminating bilateral counterparty risk "
            "through multilateral netting. The industry's move from T+2 to T+1 settlement in North America "
            "(effective 2024) compresses the post-trade window, demanding real-time matching and exception "
            "management. Settlement finality in TARGET2-Securities (T2S) leverages Delivery-versus-Payment "
            "(DvP) to ensure simultaneous exchange of cash and securities, eliminating principal risk. "
            "Intraday liquidity management monitors real-time gross settlement (RTGS) queues to avoid "
            "gridlock caused by gridlock-resolution algorithms."
        ),
    },
    {
        "id": "doc_008",
        "title": "Credit Risk Modelling: PD, LGD, EAD, and IRB Approaches",
        "content": (
            "Credit risk quantification under the Internal Ratings-Based (IRB) approach requires estimating "
            "three components: Probability of Default (PD), Loss Given Default (LGD), and Exposure at "
            "Default (EAD). PD models are typically logistic regression or machine-learning classifiers "
            "trained on historical loan performance data, with through-the-cycle versus point-in-time "
            "calibrations serving different purposes. LGD models account for collateral quality, seniority, "
            "and recovery timing; downturn LGD estimates incorporate stressed recovery rates. Credit "
            "Valuation Adjustment (CVA) extends credit risk to OTC derivatives, measuring the market value "
            "of counterparty default risk using expected positive exposure (EPE) profiles. Wrong-Way Risk "
            "arises when counterparty credit quality is negatively correlated with the value of the "
            "derivative, amplifying CVA and requiring specialist simulation techniques."
        ),
    },
    {
        "id": "doc_009",
        "title": "Basel III/IV Regulatory Capital Framework",
        "content": (
            "Basel III/IV reforms introduced a comprehensive overhaul of bank capital adequacy, imposing "
            "higher minimum Common Equity Tier 1 (CET1) ratios, a leverage ratio floor, and revised "
            "output floors that limit the benefit banks can derive from internal models relative to the "
            "standardised approach. The Fundamental Review of the Trading Book (FRTB) replaced VaR with "
            "Expected Shortfall and introduced a boundary between banking book and trading book to reduce "
            "regulatory arbitrage. The Net Stable Funding Ratio (NSFR) ensures banks fund long-term assets "
            "with stable liabilities over a one-year horizon, while the Liquidity Coverage Ratio (LCR) "
            "mandates a minimum buffer of High-Quality Liquid Assets (HQLA) to cover 30-day stressed "
            "outflows. Pillar 2 SREP assessments allow supervisors to impose institution-specific add-ons "
            "above the Pillar 1 minimum based on idiosyncratic risk profiles."
        ),
    },
    {
        "id": "doc_010",
        "title": "Quantitative Factor Models: Fama-French, Momentum, and Alternative Data",
        "content": (
            "Multi-factor equity models decompose portfolio returns into systematic risk premia and "
            "idiosyncratic alpha. The Fama-French 5-factor model extends CAPM with size (SMB), value "
            "(HML), profitability (RMW), and investment (CMA) factors. Momentum (WML) captures the "
            "empirically documented tendency of recent winners to outperform recent losers over 3-12 month "
            "horizons. Modern quant funds augment traditional factors with alternative data: satellite "
            "imagery of retailer car parks, credit-card transaction panels, and NLP-derived sentiment "
            "scores from earnings call transcripts. Covariance matrix estimation for large universes "
            "(N > 5000 stocks) uses shrinkage estimators (Ledoit-Wolf) or Random Matrix Theory (RMT) "
            "to produce well-conditioned matrices suitable for mean-variance optimisation without "
            "overfitting to sample noise."
        ),
    },
    {
        "id": "doc_011",
        "title": "Retail Order Flow, Payment for Order Flow (PFOF), and Market Impact",
        "content": (
            "Retail brokerage platforms route customer market orders to wholesale market makers "
            "(e.g., Citadel Securities, Virtu Financial) rather than directly to lit exchanges, "
            "in exchange for a per-share rebate known as Payment for Order Flow (PFOF). Because "
            "retail orders are largely uninformed — driven by sentiment rather than material "
            "non-public information — they carry minimal adverse selection risk for the market "
            "maker, who can profitably internalise them by offering a small price improvement "
            "over the NBBO (National Best Bid and Offer). The 2021 GameStop short squeeze exposed "
            "how this architecture creates systemic dependencies: Robinhood's clearing broker "
            "required a $3 billion intraday margin deposit from the NSCC, forcing the platform "
            "to restrict buy-side order flow. From an engineering standpoint, retail-facing "
            "brokers must maintain sub-100 ms order acknowledgement latency while simultaneously "
            "running real-time margin and buying-power calculations across millions of accounts, "
            "typically using in-memory data grids (Hazelcast, Redis) and event-sourced ledger "
            "architectures to ensure consistency under peak load."
        ),
    },
]