from __future__ import annotations

import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

# Ensure repo root is on sys.path so `config` and `quant` import reliably.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.settings import TRADING_DAYS  # noqa: F401
from quant.portfolio import PortfolioOptimizer
from quant.risk import RiskAnalyzer
from quant.technical import TechnicalAnalyzer
from quant import QuantEngine


def _make_yf_ohlcv(symbol: str, n_days: int = 260):
    dates = pd.bdate_range(
        end=pd.Timestamp.today().normalize(),
        periods=n_days,
    )
    np.random.seed(42)
    close = 150.0 + np.random.randn(n_days).cumsum()

    df = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, n_days).astype(float),
        }
    )
    df = df.set_index("date")
    df.index.name = "Date"
    out = df.reset_index()
    out = out.rename(columns={"Date": "date"})
    return out


@pytest.fixture
def sample_data():
    tickers = ["GOOGL", "AAPL", "META", "MSFT", "VOO"]
    prices = {t: _make_yf_ohlcv(t, n_days=504) for t in tickers}  # ~2 years
    benchmark = _make_yf_ohlcv("SPY", n_days=504)
    portfolio = [
        {
            "ticker": "GOOGL",
            "quantity": 5,
            "average_buy_price": 150.0,
            "current_price": 180.0,
            "market_value": 900.0,
        },
        {
            "ticker": "AAPL",
            "quantity": 10,
            "average_buy_price": 170.0,
            "current_price": 210.0,
            "market_value": 2100.0,
        },
        {
            "ticker": "META",
            "quantity": 3,
            "average_buy_price": 400.0,
            "current_price": 600.0,
            "market_value": 1800.0,
        },
        {
            "ticker": "MSFT",
            "quantity": 4,
            "average_buy_price": 350.0,
            "current_price": 380.0,
            "market_value": 1520.0,
        },
        {
            "ticker": "VOO",
            "quantity": 8,
            "average_buy_price": 400.0,
            "current_price": 480.0,
            "market_value": 3840.0,
        },
    ]
    return {
        "prices": prices,
        "portfolio": portfolio,
        "benchmark": benchmark,
        "news": {t: [] for t in tickers},
    }


def test_rsi_returns_valid_signal(sample_data):
    analyzer = TechnicalAnalyzer(sample_data["prices"])
    r = analyzer.calculate_rsi("GOOGL")
    assert r is not None
    assert r["signal"] in ["Overbought", "Neutral", "Oversold"]
    assert 0 <= r["rsi_current"] <= 100


def test_macd_returns_signal(sample_data):
    analyzer = TechnicalAnalyzer(sample_data["prices"])
    r = analyzer.calculate_macd("AAPL")
    assert r is not None
    assert r["signal"] in ["Bullish", "Bearish"]
    assert isinstance(r["crossover"], bool)


def test_sharpe_is_finite(sample_data):
    risk = RiskAnalyzer(
        prices=sample_data["prices"],
        benchmark=sample_data["benchmark"],
        portfolio=sample_data["portfolio"],
    )
    r = risk.sharpe_ratio()
    assert r is not None
    sharpe = r["sharpe"]
    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)
    assert not np.isinf(sharpe)


def test_max_drawdown_is_negative(sample_data):
    risk = RiskAnalyzer(
        prices=sample_data["prices"],
        benchmark=sample_data["benchmark"],
        portfolio=sample_data["portfolio"],
    )
    r = risk.max_drawdown()
    assert r is not None
    assert r["max_drawdown"] <= 0


def test_var_95_is_negative(sample_data):
    risk = RiskAnalyzer(
        prices=sample_data["prices"],
        benchmark=sample_data["benchmark"],
        portfolio=sample_data["portfolio"],
    )
    r = risk.var_95()
    assert r is not None
    assert r["var_95"] < 0


def test_monte_carlo_shape(sample_data):
    opt = PortfolioOptimizer(
        prices=sample_data["prices"],
        portfolio=sample_data["portfolio"],
    )
    r = opt.monte_carlo()
    sims = r["simulations"]
    assert sims.shape == (1000, 252)
    assert r["percentile_5"] < r["percentile_50"] < r["percentile_95"]


def test_correlation_matrix_shape(sample_data):
    opt = PortfolioOptimizer(
        prices=sample_data["prices"],
        portfolio=sample_data["portfolio"],
    )
    r = opt.correlation_matrix()
    matrix = r["matrix"]
    assert matrix.shape == (5, 5)
    assert np.all(matrix.to_numpy().diagonal() == 1.0)


def test_efficient_frontier_weights_sum_to_one(sample_data):
    opt = PortfolioOptimizer(
        prices=sample_data["prices"],
        portfolio=sample_data["portfolio"],
    )
    r = opt.efficient_frontier()
    weights = r["max_sharpe_weights"]
    total_w = sum(weights.values())
    assert np.isclose(total_w, 1.0, atol=0.01)


def test_committee_input_has_required_keys(sample_data):
    qe = QuantEngine(sample_data)
    ci = qe.get_committee_input()
    for k in ["portfolio_sharpe", "portfolio_beta", "monte_carlo", "optimal_weights", "per_ticker"]:
        assert k in ci
    per_ticker = ci["per_ticker"]
    expected = sorted(list(sample_data["prices"].keys()))
    assert sorted(list(per_ticker.keys())) == expected
