# AI Portfolio Intelligence System:
# A Quant-Validated Multi-Agent Framework for Regime-Aware Portfolio Rebalancing

**Author:** Daniel Pavelko  
**Course:** DATA-580F: AI for Decision Making  
**Institution:** Binghamton University  
**Date:** May 1, 2026

---

## Abstract
Large language models (LLMs) can generate persuasive portfolio narratives, but unconstrained outputs may conflict with quantitative risk evidence. This paper presents a production-style AI Portfolio Intelligence System that combines deterministic quantitative modeling with a three-agent LLM committee (Bull, Bear, CRO) and post-hoc mathematical validation. The framework integrates multi-source portfolio ingestion, technical and risk analytics, optional advanced engines (GARCH, HMM, HRP, RAAM, walk-forward backtesting), and human-readable reporting via dashboard and PDF. Conceptually, the committee architecture is inspired by TradingAgents (Xiao et al., 2024), while portfolio ranking extensions borrow from ranked asset allocation ideas (Giordano, 2020). Using real pipeline outputs from April 2026, we show the system can produce coherent rebalancing proposals, quantify portfolio risk posture, and detect weak recommendation quality through consistency and citation audits. Results highlight a central finding: multi-agent agreement on asset picks does not guarantee agreement on risk levels, motivating explicit verification layers in financial AI systems.

**Keywords:** multi-agent LLM, portfolio analytics, risk management, regime detection, HRP, RAAM, auditability

---

## 1. Introduction
Retail and semi-professional portfolio workflows are often fragmented: brokerage data retrieval, quantitative analysis, and decision interpretation are handled in different tools with weak traceability across steps. In parallel, LLM-based advisory systems have improved natural-language synthesis but can produce outputs that are difficult to validate quantitatively.

This work addresses that gap with a quant-first architecture where:
1. deterministic models compute risk/return signals first,
2. specialized LLM agents debate on top of that evidence, and
3. post-rebalance validation re-checks claims numerically.

The system is designed as decision support, not automated execution. Its emphasis is methodological reliability: evidence-backed recommendations, explicit assumptions, and reproducible outputs.

### 1.1 Research Motivation
The project tackles a real-world problem: portfolio decisions under uncertainty with mixed quality data, incomplete model confidence, and conflicting growth-vs-risk objectives. The design goal is not highest possible return, but a transparent and auditable process for balancing downside protection and opportunity capture.

### 1.2 Contributions
This paper makes four practical contributions:
1. **Quant-validated multi-agent pipeline:** A Bull/Bear/CRO architecture where recommendations are checked against computed metrics instead of accepted at face value.
2. **Hybrid modeling stack:** Integration of technical, risk, regime, volatility, optimization, and ranking signals into a unified committee input schema.
3. **Audit layer for AI reliability:** Consistency and citation-quality scoring to measure recommendation stability and evidence alignment.
4. **End-to-end reproducibility artifacts:** Structured JSON run cache, dashboard panels, and PDF tear sheet enabling run-to-run comparison.

---

## 2. Related Work

### 2.1 Multi-Agent LLMs in Finance
TradingAgents (Xiao et al., 2024) demonstrates that role-specialized agents can improve financial decision generation versus monolithic prompting. Our system adopts this spirit (specialized roles and arbitration) but adds a strict post-rebalance validator and reliability audits.

### 2.2 Portfolio Construction and Risk
Classical portfolio theory (Markowitz, 1952) motivates return-risk optimization but can be unstable under noisy covariance estimation. HRP (López de Prado, 2016) addresses this through hierarchical clustering and risk parity logic.

### 2.3 Volatility and Regime Modeling
GARCH (Engle, 1982) captures volatility clustering; regime-switching paradigms (Hamilton, 1989) motivate HMM-based market-state inference. In this project, regime outputs modulate risk posture rather than directly predict returns.

### 2.4 Ranked Allocation Extensions
Ranked asset allocation frameworks (Giordano, 2020) provide interpretable ranking-based selection logic. The RAAM module in this project adapts that philosophy with regime-conditional factor weighting and cash substitution when momentum fails.

---

## 3. System Architecture
The system is implemented as a layered pipeline:

1. **Data Layer**
   - Portfolio sources: Robinhood API, CSV/Excel imports, screenshot extraction, manual entry.
   - Price source: yfinance.
   - Persistence: SQLite + SQLAlchemy.

2. **Quantitative Layer**
   - Technical metrics: RSI, MACD, SMA.
   - Portfolio metrics: Sharpe, VaR/CVaR, beta, alpha, drawdown.
   - Optimization context: HRP/evidence schema.

3. **Advanced Quant Layer (configurable)**
   - HMM regime engine.
   - GARCH volatility forecasts.
   - Trend regression signals.
   - Walk-forward backtest engine.
   - RAAM ranking engine.

4. **AI Committee Layer**
   - Bull agent (upside thesis),
   - Bear agent (risk thesis),
   - CRO agent (allocation arbitration).

5. **Validation and Reporting Layer**
   - Post-rebalance metric checks.
   - Consistency audit across repeated committee runs.
   - Citation quality audit.
   - Dashboard and PDF output.

---

## 4. Methodology

### 4.1 Quant-First Committee Input Design
The committee does not consume raw narratives. It receives a structured bundle:
1. portfolio summary and exposures,
2. regime probabilities and risk scalar,
3. per-ticker signals (trend/risk/technical),
4. optimization constraints/signals (including RAAM when enabled),
5. low-confidence flags.

This forces agent reasoning to remain anchored to measurable quantities.

### 4.2 Multi-Agent Arbitration Protocol
The protocol is sequential:
1. Bull submits pro-growth recommendations.
2. Bear submits risk/counter recommendations.
3. CRO reconciles with constraints and outputs target weights summing to 100%.

This process operationalizes adversarial reasoning while preserving a single final allocation surface.

### 4.3 Reliability Controls
Two reliability checks are included:
1. **Consistency audit:** repeat identical committee runs and measure variance in risk score, weights, and top-ticker agreement.
2. **Citation-quality audit:** classify cited evidence as consistent/contradictory/ambiguous by agent role.

### 4.4 Backtesting and Regime-Aware Comparison
When enabled, walk-forward validation compares strategy behavior over rolling windows. The project also supports strategy-level comparisons that can include HRP+regime, equal-weight benchmark, and RAAM variants.

---

## 5. Experimental Setup

### 5.1 Runtime Environment
The system is run through `main.py` with cached outputs in `outputs/*.json`.

### 5.2 Data Sample Used in This Paper
Primary metrics were extracted from:
1. `outputs/latest_run.json` (timestamp: 2026-04-22 20:02),
2. `outputs/run_20260418_220540.json` (contains full backtest block),
3. `outputs/consistency_audit.json`,
4. `outputs/citation_audit.json`.

### 5.3 Evaluation Dimensions
We evaluate:
1. portfolio/risk context quality,
2. committee output coherence,
3. backtest validity metadata,
4. reliability diagnostics (consistency and citation quality).

---

## 6. Results

### 6.1 Portfolio and Regime Snapshot (Latest Run)
From `latest_run.json`:
1. Portfolio value: **$1,171.22**
2. Position count: **15**
3. Regime probabilities: **Bull 0.457, Neutral 0.370, Bear 0.173**
4. Dominant regime: **Bull**
5. Risk scalar: **0.358**
6. Portfolio Sharpe: **1.3634**
7. Portfolio beta: **1.1938**
8. Portfolio alpha: **0.0746**

Committee output:
1. Portfolio risk score: **6/10**
2. Final target table size: **16 positions**
3. Example warning flags:
   - Sharpe degradation after rebalance: **34.6%**
   - Monte Carlo 5th percentile improvement only: **3.0%**

Interpretation: the validator correctly surfaces that a defensive rebalance can reduce risk but may underperform on risk-adjusted return.

### 6.2 RAAM Signal Example (Latest Run)
RAAM committee input block reported:
1. selected tickers: `['VXUS', 'QQQ', 'VGT', 'IAU', 'TQQQ']`
2. weights: `20%` each (equal among selected assets)

This demonstrates that ranked-selection logic can be exported as a compact signal set for committee deliberation.

### 6.3 Backtest Block Evidence
From `run_20260418_220540.json`:
1. backtest periods: **15**
2. benchmark CAGR: **0.0946**
3. HRP Sharpe: **0.0874**
4. benchmark Sharpe: **0.5234**
5. regime timing value: **-0.0365**

Not all newer fields were present in this historical run schema, but the output still demonstrates periodized walk-forward evaluation with comparative metrics.

### 6.4 Reliability Diagnostics
From `consistency_audit.json`:
1. runs: **1** successful run out of 1
2. overall consistency score: **70.0/100**
3. risk score standard deviation: **0.0**
4. cash weight CV: **0.0**

From `citation_audit.json`:
1. Bull citation quality grade: **D**, contradiction rate **45.5%** (11 citations)
2. Bear citation quality grade: **D**, contradiction rate **50.0%** (10 citations)

Interpretation: recommendation generation can remain operational while evidence citation quality is weak. This justifies keeping citation auditing as a first-class control.

---

## 7. Discussion

### 7.1 What Worked
The architecture successfully integrates:
1. heterogeneous data inputs,
2. quantitative analytics,
3. role-specialized AI synthesis,
4. explicit warning/validation surfaces.

The most important practical gain is traceability: each recommendation can be traced back to measurable inputs and post-check outputs.

### 7.2 Main Insight
A critical insight from audits is that **selection agreement and risk agreement are different problems**. Even when agent recommendations look coherent, citation contradiction rates can remain high, indicating weak evidence alignment.

### 7.3 Relationship to TradingAgents
Compared with TradingAgents-style role decomposition, this project contributes an additional deterministic validation spine and explicit quality audits. In practice, this helps move from “plausible narrative” to “quantified recommendation quality.”

---

## 8. Threats to Validity
1. Some cached runs were generated under older schema versions, limiting field comparability across dates.
2. Reliability audit shown here used one run; stronger inference requires larger `n` (for example 5-20).
3. Brokerage/API behavior and missing-data fallbacks can alter exposure context between runs.
4. Backtest quality is sensitive to horizon depth and asset history coverage.

---

## 9. Conclusion
This paper presents a deployable research-grade framework for AI-assisted portfolio rebalancing that is quant-first, role-structured, and auditable. Empirical run artifacts show the system can produce coherent outputs while also exposing where LLM reasoning is weak (for example citation contradictions), which is essential for trustworthy financial AI workflows.

The central methodological takeaway is that multi-agent orchestration should be paired with deterministic verification and explicit quality diagnostics, not treated as a standalone intelligence layer.

---

## 10. Future Work
1. Run higher-`n` consistency studies with temperature-controlled sampling.
2. Add statistical significance testing across strategy variants (HRP/RAAM/EW).
3. Expand citation parser to deeper semantic grounding with quantitative cross-checks.
4. Incorporate transaction costs and slippage in backtest realism.
5. Add standardized benchmark datasets for cross-run reproducibility.

---

## References
Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica, 50*(4), 987-1007.

Giordano, G. (2020). *Ranked Asset Allocation*.

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica, 57*(2), 357-384.

López de Prado, M. (2016). Building diversified portfolios that outperform out of sample. *The Journal of Portfolio Management, 42*(4), 59-69.

Markowitz, H. (1952). Portfolio selection. *The Journal of Finance, 7*(1), 77-91.

Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. *The Journal of Risk, 2*(3), 21-41.

Xiao, Y., Sun, E., Luo, D., & Wang, W. (2024). TradingAgents: Multi-Agents LLM financial trading framework. *arXiv preprint arXiv:2412.20138*. https://arxiv.org/abs/2412.20138

