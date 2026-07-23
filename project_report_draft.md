# AI Portfolio Intelligence System
## Regime-Aware Quantitative Portfolio Analysis with a Multi-Agent LLM Committee

**Course:** DATA-580F: AI for Decision Making  
**Student:** Daniel Pavelko  
**University:** Binghamton University  
**Instructor:** [Instructor Name]  
**Semester:** [Semester, Year]  
**Submission Date:** [Month Day, Year]

---

## Formatting Assumptions
This draft is prepared to fit a typical graduate project-report format:
1. 8-12 pages main body.
2. 12 pt font, double-spaced, 1-inch margins.
3. Numbered sections.
4. Figures/tables with captions.
5. APA-style references.

If your official template differs, keep the content and map headings/spacing to your required format.

---

## Abstract
This project presents an end-to-end AI Portfolio Intelligence System that combines quantitative finance models with a structured multi-agent large language model (LLM) committee for portfolio decision support. The pipeline ingests portfolio holdings, retrieves market data, computes technical and risk metrics, detects market regimes, optimizes allocation, and generates validated rebalancing recommendations. Quantitative components include RSI/MACD/SMA indicators, Value-at-Risk and Conditional Value-at-Risk, Monte Carlo simulation, GARCH(1,1) volatility modeling, Hidden Markov Model (HMM) regime detection, Hierarchical Risk Parity (HRP) optimization, and walk-forward backtesting. A three-agent committee (Bull, Bear, CRO) debates recommendations using shared quantitative evidence; the CRO output is then mathematically re-validated. Outputs are delivered through a Streamlit dashboard and a multi-page PDF tear sheet. Results indicate improved risk-adjusted positioning and reduced downside exposure in stressed regimes while preserving transparency through auditable model outputs and citation checks. The system is intended for educational and research use and demonstrates a practical architecture for AI-assisted portfolio analytics under real-world data constraints.

**Keywords:** portfolio optimization, HMM, GARCH, HRP, walk-forward backtesting, multi-agent LLM, risk management

---

## 1. Introduction
Portfolio management requires balancing return targets with drawdown control, diversification, and changing market conditions. In many practical workflows, these tasks are fragmented across separate tools for data collection, analytics, and interpretation. This project addresses that gap by integrating all stages into a single, reproducible AI-assisted pipeline.

The key design principle is quantitative-first reasoning. Rather than allowing an LLM to generate unconstrained recommendations, the system computes deterministic financial and risk features first, then passes this evidence to a structured three-agent committee. A final post-rebalance validation stage re-runs metrics on proposed allocations to verify whether committee claims are numerically supported.

This architecture supports three goals:
1. Improve robustness through regime-aware risk controls.
2. Maintain explainability via explicit model outputs and debate traces.
3. Increase reliability through deterministic post-hoc validation.

---

## 2. Problem Statement and Project Objectives
The project focuses on decision support for portfolio analysis and rebalancing under uncertain market conditions. The central challenge is combining statistical rigor and human-interpretable recommendations without sacrificing reproducibility.

### 2.1 Objectives
1. Ingest holdings from live brokerage and offline import sources into a unified schema.
2. Compute portfolio and per-ticker technical, volatility, and downside-risk features.
3. Detect market regime and encode it into actionable risk posture.
4. Generate allocation recommendations using optimization and committee arbitration.
5. Validate recommendations against objective post-rebalance metrics.
6. Provide clear outputs through dashboard and formal report artifacts.

### 2.2 Scope and Boundaries
In scope are analytics, recommendation generation, and reporting. Out of scope are automatic trade execution, broker order routing, and any claim of investment advice.

---

## 3. System Architecture
The system is organized into four functional layers:

1. **Data Layer**  
Sources: Robinhood API, yfinance, CSV/Excel import, screenshot extraction, and manual entry.  
Storage and persistence: SQLite through SQLAlchemy ORM.

2. **Quantitative Engine**  
Technical indicators, portfolio risk metrics, Monte Carlo simulation, and optimization context.

3. **Advanced Modeling Layer**  
GARCH volatility forecasting, HMM regime detection, trend analysis, and walk-forward backtesting.

4. **AI Committee + Reporting Layer**  
Bull/Bear/CRO arbitration, post-rebalance validator, Streamlit dashboard, and PDF tear sheet generation.

**Figure 1 (Placeholder):** End-to-end architecture diagram.  
`[Insert architecture image from README or generated system diagram]`

---

## 4. Data Pipeline and Preprocessing
The portfolio ingestion module supports both live and offline workflows. Live mode pulls positions from brokerage APIs; offline mode imports broker exports and screenshot-derived positions through a normalization interface. This multi-source design improves practicality and reproducibility when brokerage auth is unavailable.

Historical prices are fetched from yfinance and stored locally. The pipeline includes fallback logic so analytics can proceed using cached or imported positions if real-time brokerage login fails. This was an important engineering requirement due to intermittent API/challenge behavior during development.

Preprocessing steps include ticker normalization, schema harmonization, null handling, and basic quality checks before downstream model execution.

---

## 5. Methodology

### 5.1 Technical and Risk Metrics
The system computes:
1. RSI, MACD, SMA50/SMA200 for technical state.
2. Sharpe and Sortino for risk-adjusted return.
3. VaR(95%) and CVaR for tail risk.
4. Max drawdown and beta for downside and market sensitivity.
5. Concentration and exposure diagnostics.

### 5.2 Volatility Modeling: GARCH(1,1)
GARCH captures time-varying volatility and clustering effects in returns. Forecasted volatility is used for risk signaling and committee context, especially in elevated stress regimes.

### 5.3 Regime Detection: Hidden Markov Model
A three-state Gaussian HMM estimates probabilities for Bull, Neutral, and Bear regimes. These probabilities map into a risk scalar used for risk-aware recommendation constraints.

### 5.4 Allocation Engine: Hierarchical Risk Parity
HRP is used to produce diversified risk-sensitive allocations based on hierarchical clustering of correlations, reducing instability common in direct covariance inversion methods.

In addition, the project includes a Ranked Asset Allocation Model (RAAM)-style ranking layer inspired by Giordano’s ranked allocation framing, used as a comparative or complementary signal for strategy selection and weighting logic.

### 5.5 Trend Engine
A regression-based trend model estimates slope, acceleration, and confidence to provide directional context complementary to volatility and regime features.

### 5.6 Walk-Forward Backtesting
A rolling train/test walk-forward framework is used to reduce look-ahead bias and evaluate out-of-sample behavior relative to benchmark alternatives.

### 5.7 Multi-Agent Committee
1. **Bull Agent:** generates upside-focused thesis.
2. **Bear Agent:** generates risk-focused counter-thesis.
3. **CRO Agent:** arbitrates with policy constraints and returns final target weights.

The committee design is conceptually aligned with recent multi-agent financial LLM frameworks, particularly TradingAgents, where specialized agents collaborate (and disagree) before final action synthesis (Xiao et al., 2024).

### 5.8 Post-Rebalance Validation
The system recomputes risk/return metrics on CRO-proposed allocations and flags unsupported claims. This validation step is central to trustworthiness.

---

## 6. Implementation Details
The codebase is modular and production-oriented:
1. `data/` handles ingestion, import normalization, database operations, validation.
2. `quant/` contains technical, risk, optimization, and advanced model engines.
3. `ai/` contains committee orchestration, prompt logic, and consistency/citation audits.
4. `reporting/` generates Streamlit visual analytics and PDF output.
5. `main.py` orchestrates full pipeline execution through CLI flags.

**Table 1 (Placeholder):** Module-to-function mapping.  
`[Insert concise table: file, input, output, purpose]`

---

## 7. Experimental Setup
All experiments are reproducible through CLI configuration.

1. Historical lookback selectable (`1y`, `2y`, `5y`).
2. Benchmark based on SPY index history.
3. Walk-forward evaluation with rolling windows.
4. Optional committee consistency audit over repeated runs.

Example commands:
```bash
python main.py --price-period 5y
python main.py --skip-committee --skip-pdf
streamlit run reporting/dashboard.py
```

---

## 8. Results
Representative outputs show measurable improvements in risk-adjusted profile after committee-guided rebalancing:
1. Sharpe ratio improvement relative to baseline portfolio.
2. Lower maximum drawdown and reduced beta exposure.
3. Improved left-tail Monte Carlo outcomes.
4. Regime-aware tradeoff where lower upside may be accepted for stronger capital preservation in Bear conditions.

Backtesting outputs include period-level diagnostics and aggregate metrics, with explicit validity framing based on number of test windows.

**Figure 2 (Placeholder):** Before/after metric comparison (Sharpe, drawdown, beta, VaR).  
**Figure 3 (Placeholder):** Walk-forward cumulative return curve vs benchmark.  
**Figure 4 (Placeholder):** Regime timeline with risk scalar history.

---

## 9. Discussion
The project demonstrates that LLM-based investment reasoning is materially stronger when bounded by deterministic quantitative context and verification. The Bull/Bear/CRO structure surfaces opposing hypotheses before final allocation, improving interpretability and reducing one-sided narrative bias.

Regime-aware controls are especially useful in adverse environments where downside mitigation can be more important than maximizing nominal return. The architecture is therefore best interpreted as a robust decision-support framework rather than a prediction engine.

---

## 10. Limitations
1. Brokerage authentication/session behavior can be unstable.
2. Backtest confidence depends on available historical depth and window count.
3. LLM outputs are stochastic and sensitive to sampling/context.
4. Data-source inconsistency can affect metric reliability.
5. No direct execution or transaction-cost-aware routing is included.

---

## 11. Conclusion
This project delivers a practical, auditable architecture for AI-assisted portfolio intelligence by combining statistical finance methods, regime awareness, robust allocation logic, and constrained multi-agent reasoning. The key contribution is not any single model but the integration pattern: quantitative evidence first, adversarial AI synthesis second, deterministic validation third.

The resulting system is reproducible, interpretable, and suitable as a research-oriented capstone demonstration of applied AI for financial decision support.

---

## 12. Future Work
1. Expand multi-strategy comparative backtesting and statistical significance testing.
2. Add live regime-transition alerting and monitoring workflows.
3. Improve broker abstraction and session resilience.
4. Add optional trade-execution adapters with strict controls and audit logging.
5. Add CI-based model regression checks and scenario stress testing.

---

## Bonus Option Positioning
This report is intentionally structured in research-paper format (problem, method, experiment, results, discussion, limitations), which supports bonus option (a): project-to-paper conversion with minimal additional engineering risk.

---

## References (APA Style Starter)
Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica, 50*(4), 987-1007.

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica, 57*(2), 357-384.

López de Prado, M. (2016). Building diversified portfolios that outperform out of sample. *The Journal of Portfolio Management, 42*(4), 59-69.

Markowitz, H. (1952). Portfolio selection. *The Journal of Finance, 7*(1), 77-91.

Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. *The Journal of Risk, 2*(3), 21-41.

Giordano, G. (2020). *Ranked Asset Allocation*.

Xiao, Y., Sun, E., Luo, D., & Wang, W. (2024). TradingAgents: Multi-Agents LLM financial trading framework. *arXiv preprint arXiv:2412.20138*. https://arxiv.org/abs/2412.20138

Streamlit Documentation. (n.d.). https://docs.streamlit.io/

SQLAlchemy Documentation. (n.d.). https://docs.sqlalchemy.org/

Plotly Python Documentation. (n.d.). https://plotly.com/python/

yfinance on PyPI. (n.d.). https://pypi.org/project/yfinance/

Anthropic Documentation. (n.d.). https://docs.anthropic.com/
