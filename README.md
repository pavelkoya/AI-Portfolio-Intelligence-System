# AI Portfolio Intelligence System

> Institutional-grade portfolio analysis combining quantitative finance with multi-agent AI — built on a real Robinhood portfolio.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Anthropic](https://img.shields.io/badge/Claude-Sonnet_4.5-CC785C?style=flat-square&logo=anthropic&logoColor=white)](https://anthropic.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=flat-square&logo=sqlite&logoColor=white)](https://sqlite.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

---

## What This Does

This system ingests a live brokerage portfolio, runs it through a stack of quantitative models, then convenes a **three-agent AI investment committee** (Bull, Bear, CRO) that debates the positions and produces a mathematically validated rebalancing recommendation — all in a single command.

```bash
python main.py
# → Fetches live portfolio from Robinhood
# → Runs GARCH, HMM, HRP, trend regression
# → Calls Claude API 3× (Bull → Bear → CRO)
# → Validates CRO claims against actual math
# → Generates PDF tear sheet + Streamlit dashboard
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                │
│  Robinhood API  ──►  yfinance  ──►  SQLite  ──►  Validator   │
│  (live portfolio)   (prices)    (4 tables)   (quality checks)│
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│                 QUANTITATIVE ENGINE                          │
│                                                              │
│  Technical      Risk Metrics      Portfolio Optimization     │
│  RSI / MACD     Sharpe / Sortino  Efficient Frontier         │
│  SMA 50/200     VaR 95% / CVaR    Monte Carlo (1000 paths)   │
│  Beta / Alpha   Max Drawdown      HRP (Hierarchical          │
│                                       Risk Parity)           │
├──────────────────────────────────────────────────────────────┤
│                  ADVANCED MODELS                             │
│                                                              │
│  GARCH(1,1)         HMM (3-state)       Linear Regression    │
│  Volatility         Market Regime       Trend Engine         │
│  Forecasting        Bull/Bear/Neutral   Slope + Acceleration │
│  10-day horizon     risk_scalar 0→1     Confidence Score     │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│              MULTI-AGENT AI COMMITTEE                        │
│                                                              │
│    Bull Agent            Bear Agent            CRO Agent     │
│  Growth-focused         Risk-focused          Arbitrates     │
│  Cites analyst          Cites CVaR            Applies        │
│  upside + momentum      + GARCH stress        regime rules   │
│  + trend signals        + concentration       Validates math │
│                                                              │
│  ──────────────────────────────────────────────────────────  │
│  Post-Rebalance Validator: reruns VaR / Sharpe / Drawdown    │
│  on proposed weights — proves or disproves CRO claims        │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│                      OUTPUT LAYER                            │
│  Streamlit Dashboard   PDF Tear Sheet   SQLite History       │
│  5 interactive panels  4-page report    Daily regime log     │
│  Live regime gauge     Dark theme       Per-ticker metrics   │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Features

### Quantitative Models
| Model | Purpose | Output |
|---|---|---|
| **GARCH(1,1)** | Time-varying volatility | 10-day vol forecast + regime (Low/Elevated/Stress) |
| **GaussianHMM** | Market regime detection | Bull/Neutral/Bear probability + risk scalar |
| **HRP** | Portfolio optimization | Hierarchical risk-weighted allocation |
| **Monte Carlo** | Scenario simulation | 1,000 paths × 252 days, 5th/50th/95th percentile |
| **Linear Regression** | Trend signals | Slope, acceleration, uncertainty, seasonal component |
| **Walk-Forward Backtest** | Strategy validation | HRP+Regime vs equal-weight benchmark, 15–20 periods |

### Multi-Agent Committee
Three Claude agents each receive the full quantitative context and debate sequentially:

- **Bull Agent** — makes the case for highest-conviction longs, citing analyst upside %, RSI oversold signals, HRP underweights, and trend conditions
- **Bear Agent** — identifies concentration risk, GARCH stress regimes, HRP REDUCE signals, positions trading above analyst targets
- **CRO Agent** — arbitrates, applies regime-based rules (risk_scalar > 0.7 → mandatory 20% cash), produces final position table with weights summing to exactly 100%

**Post-rebalance validation** reruns VaR, Sharpe, Max Drawdown, Beta, and Monte Carlo on the proposed weights and flags if CRO claims are not mathematically supported.

### Portfolio Import
Supports four import methods with a unified edit interface:
- **Robinhood API** — live positions via robin_stocks
- **CSV / Excel** — auto-detects 6 brokers (Robinhood, Fidelity, Schwab, IBKR, Freedom Broker, Tinkoff) including Russian column names
- **Screenshot** — Claude Vision or Gemini Flash extracts positions from any brokerage UI screenshot in any language
- **Manual entry** — add positions by shares, dollar amount, or target weight %

---

## Tech Stack

```
Language:       Python 3.9
AI:             Anthropic Claude claude-sonnet-4-5 (3-agent pipeline)
                Google Gemini Flash (screenshot OCR, optional)
Quant:          arch (GARCH), hmmlearn (HMM), PyPortfolioOpt (HRP/EF)
                pandas-ta (RSI/MACD), scipy (regression)
Data:           robin_stocks, yfinance, Financial Modeling Prep API
Storage:        SQLite via SQLAlchemy ORM (4 tables, schema migrations)
Dashboard:      Streamlit + Plotly
PDF:            reportlab (4-page institutional tear sheet)
```

---

## Project Structure

```
portfolio_ai/
├── main.py                     # Single entry point — full pipeline
├── config/
│   ├── settings.py             # Environment config + constants
│   └── broker_maps.py          # Broker column mapping registry
├── data/
│   ├── data_loader.py          # Robinhood + yfinance ingestion
│   ├── database.py             # SQLAlchemy ORM + migrations
│   ├── validator.py            # Data quality checks
│   ├── analyst_fetcher.py      # yfinance analyst targets
│   ├── csv_importer.py         # Multi-broker CSV/Excel parser
│   └── screenshot_importer.py  # Vision AI portfolio extractor
├── quant/
│   ├── technical.py            # RSI, MACD, SMA
│   ├── risk.py                 # Sharpe, VaR, Drawdown, Beta
│   ├── portfolio.py            # Monte Carlo, Efficient Frontier
│   ├── garch_engine.py         # GARCH(1,1) vol forecasting
│   ├── regime_engine.py        # HMM market regime detection
│   ├── hrp_engine.py           # Hierarchical Risk Parity
│   ├── trend_engine.py         # Linear regression trend signals
│   ├── risk_levels.py          # ATR stop-loss calculator
│   ├── post_rebalance_engine.py# CRO claim validator
│   └── backtest_engine.py      # Walk-forward backtester
├── ai/
│   ├── committee_inputs.py     # Input assembler for all agents
│   ├── committee.py            # Multi-agent orchestrator
│   └── prompts.py              # All agent prompt templates
├── reporting/
│   ├── dashboard.py            # Streamlit app (5 panels)
│   ├── pdf_generator.py        # reportlab tear sheet
│   └── pages/
│       └── import_portfolio.py # Portfolio import UI
└── tests/
    ├── test_data_loader.py
    └── test_quant.py
```

---

## Quick Start

**1. Clone and install**
```bash
git clone https://github.com/yourusername/portfolio-ai.git
cd portfolio-ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**2. Configure environment**
```bash
cp .env.example .env
# Add your keys:
# ROBINHOOD_USERNAME=your@email.com
# ROBINHOOD_PASSWORD=yourpassword
# ANTHROPIC_API_KEY=sk-ant-...
```

**3. Run the pipeline**
```bash
# Full run (fetches live portfolio, runs all models, calls AI)
python main.py

# Skip AI committee (test data pipeline only)
python main.py --skip-committee --skip-pdf

# Use cheaper model for testing
python main.py --model openai-gpt4o-mini

# Use cached data (no yfinance re-fetch)
python main.py --no-refresh
```

**4. Launch dashboard**
```bash
streamlit run reporting/dashboard.py
```

**5. Import from CSV or screenshot**
```bash
streamlit run reporting/pages/import_portfolio.py
```

---

## CLI Reference

```
python main.py [options]

  --model           LLM to use: anthropic (default), anthropic-opus,
                    openai-gpt4o, openai-gpt4o-mini
  --skip-committee  Skip all Claude API calls (data pipeline only)
  --skip-pdf        Skip PDF generation
  --skip-backtest   Skip walk-forward backtest
  --no-refresh      Use existing DB data, skip yfinance download
  --price-period    Historical data: 1y, 2y (default), 5y
  --output-dir      Output directory (default: outputs/)
```

---

## Dashboard Panels

| Panel | Contents |
|---|---|
|  **Regime** | HMM risk scalar gauge, Bull/Bear/Neutral probabilities, trend signals table |
|  **Portfolio Health** | Current vs HRP weights chart, rebalancing table (ADD/REDUCE/HOLD), before/after validation |
|  **Risk Levels** | Per-ticker stop loss, take profit, ATR, GARCH vol regime, RSI, SMA200 |
|  **Analyst Context** | Analyst price targets, upside %, BUY/HOLD/REDUCE signals |
|  **Committee** | Bull/Bear debate, CRO verdict, risk score (1–10), executive summary |

The sidebar includes a **Generate PDF Report** button that produces a 4-page institutional tear sheet.

---

## Sample Output

```
╔══════════════════════════════════╗
║  AI PORTFOLIO INTELLIGENCE       ║
║  Pipeline Complete               ║
╚══════════════════════════════════╝
Total runtime:  187.3s
Risk Score:     9/10
Regime:         Bear (scalar=0.957)
Validation:     PASSED — all checks clear
  Sharpe:       0.452 → 0.970  (+115%)
  Max Drawdown: -21.3% → -13.4%  (-37%)
  Beta:         1.084 → 0.685  (-37%)
  MC 5th Pct:  -21.9% → -6.5%  (+15pp)
```

---

## Academic Context

Built as the capstone project for **DATA-580F: AI for Decision Making** at Binghamton University. The system demonstrates end-to-end application of probabilistic models, reinforcement of quantitative finance theory, and production-grade AI engineering.

**Models covered in coursework:**
- Hidden Markov Models (regime detection)
- GARCH family (volatility clustering)
- Portfolio optimization theory (Markowitz → HRP)
- Monte Carlo simulation
- Walk-forward validation (avoiding look-ahead bias)
- Multi-agent LLM architectures

---

## Roadmap

- [ ] Live price WebSocket feed (replace polling)
- [ ] Options chain analysis layer
- [ ] Sector rotation signals
- [ ] Email/Slack alert system for regime changes
- [ ] Historical regime overlay on price charts
- [ ] IBKR API integration (live trading execution)

---

## License

MIT — see [LICENSE](LICENSE)

---

## Author

**Daniel Pavelko** — Graduate Student, Binghamton University  
Quantitative Finance | AI Engineering | Data Science

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/pavelkoya)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/pavelkoya)

---

> ⚠️ *This system is for educational and research purposes. Nothing in this repository constitutes investment advice.*
