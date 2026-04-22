from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from config.settings import BENCHMARK_TICKER
from data.data_loader import DataLoader
from data.database import DatabaseManager
from data.validator import DataValidator


def _make_yf_ohlcv(symbol: str, n_days: int = 260):
    dates = pd.date_range(
        end=pd.Timestamp.today(),
        periods=n_days,
        freq="B",  # business days only
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


def _make_positions() -> Dict[str, Dict[str, Any]]:
    # robin_stocks build_holdings() shape (subset of keys used by DataLoader).
    return {
        "AAPL": {
            "type": "equity",
            "quantity": 2,
            "average_buy_price": 150.0,
            "price": 190.0,
            "equity": 380.0,
            "equity_change": 40.0,
            "percent_change": 26.67,
        },
        "MSFT": {
            "type": "equity",
            "quantity": 3,
            "average_buy_price": 250.0,
            "price": 305.0,
            "equity": 915.0,
            "equity_change": 165.0,
            "percent_change": 22.0,
        },
        "BTC": {
            "type": "crypto",
            "quantity": 0.1,
            "average_buy_price": 42000.0,
            "price": 50000.0,
            "equity": 5000.0,
            "equity_change": 800.0,
            "percent_change": 19.0,
        },
    }


@pytest.fixture
def loader_with_tmp_db(tmp_path):
    db_path = tmp_path / "test_portfolio_ai.db"
    db = DatabaseManager(db_path=str(db_path))
    return DataLoader(db_manager=db)


def test_portfolio_structure(loader_with_tmp_db):
    loader = loader_with_tmp_db
    loader._robinhood_logged_in = True  # Avoid Robinhood login in unit tests.

    with patch(
        "robin_stocks.robinhood.account.build_holdings", autospec=True
    ) as mock_build_holdings:
        mock_build_holdings.return_value = _make_positions()
        positions = loader.get_portfolio()

    assert isinstance(positions, list)
    assert len(positions) == 3

    expected_keys = {
        "ticker",
        "quantity",
        "average_buy_price",
        "current_price",
        "market_value",
        "unrealized_pnl",
        "unrealized_pnl_pct",
        "equity_type",
    }
    for pos in positions:
        assert expected_keys.issubset(set(pos.keys()))
        assert isinstance(pos["ticker"], str)
        assert pos["ticker"] in {"AAPL", "MSFT", "BTC"}

    latest = loader.db.get_latest_portfolio()
    assert not latest.empty
    assert len(latest) == 3


def test_historical_prices_yfinance():
    loader = DataLoader(db_manager=DatabaseManager(db_path=":memory:"))

    # Real yfinance call for 1 month of META.
    import yfinance as yf

    df_raw = yf.download(
        "META",
        period="1mo",
        interval="1d",
        progress=False,
        auto_adjust=False,
        actions=False,
        group_by="column",
    )
    if df_raw is None or df_raw.empty:
        pytest.skip("yfinance returned empty data (network/offline environment)")

    df = loader._normalize_yfinance_ohlcv(df_raw)
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert not df.empty
    assert df["close"].isna().sum() == 0


def test_validator_catches_empty_df():
    validator = DataValidator()
    ok, errors = validator.validate_prices(pd.DataFrame(), "TEST")
    assert ok is False
    assert errors and isinstance(errors, list)


def test_validator_cleans_nulls():
    validator = DataValidator()

    # Create 10 days where 'close' has 3 consecutive nulls that should be forward-filled.
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": [100 + (10 * i / 9) for i in range(10)],
            "high": [101 + (10 * i / 9) for i in range(10)],
            "low": [99 + (10 * i / 9) for i in range(10)],
            "close": [100 + (10 * i / 9) for i in range(10)],
            "volume": [1_000_000 for _ in range(10)],
        }
    )
    df.loc[2:4, "close"] = None  # exactly 3 consecutive nulls

    cleaned = validator.clean_prices(df)
    assert cleaned["close"].isna().sum() == 0


def test_full_pipeline_mock(tmp_path):
    db_path = tmp_path / "test_portfolio_ai_pipeline.db"
    loader = DataLoader(db_manager=DatabaseManager(db_path=str(db_path)))
    validator = DataValidator()

    # Mock Robinhood login, holdings, and news.
    fake_holdings = {
        "AAPL": {
            "type": "equity",
            "quantity": 2,
            "average_buy_price": 150.0,
            "price": 190.0,
            "equity": 380.0,
            "equity_change": 40.0,
            "percent_change": 26.67,
        },
        "MSFT": {
            "type": "equity",
            "quantity": 3,
            "average_buy_price": 250.0,
            "price": 305.0,
            "equity": 915.0,
            "equity_change": 165.0,
            "percent_change": 22.0,
        },
    }

    fake_news_items = [
        {
            "title": "Headline 1",
            "source": "UnitTest",
            "published_at": "2026-03-01T12:00:00Z",
        },
        {
            "title": "Headline 2",
            "source": "UnitTest",
            "published_at": "2026-03-02T12:00:00Z",
        },
    ]

    price_dfs = {
        "AAPL": _make_yf_ohlcv("AAPL"),
        "MSFT": _make_yf_ohlcv("MSFT"),
        "SPY": _make_yf_ohlcv("SPY"),
    }

    with (
        patch("robin_stocks.robinhood.authentication.login", autospec=True) as mock_login,
        patch("robin_stocks.robinhood.account.build_holdings", autospec=True) as mock_holdings,
        patch("robin_stocks.robinhood.stocks.get_news", autospec=True) as mock_get_news,
        patch.object(
            loader,
            "get_historical_prices",
            return_value=price_dfs,
        ),
    ):
        mock_login.return_value = {"access_token": "fake-token"}  # non-None => success
        mock_holdings.return_value = fake_holdings
        mock_get_news.return_value = fake_news_items

        raw = loader.get_all_data()
        cleaned = validator.validate_all(raw)

    assert set(cleaned.keys()) == {"portfolio", "prices", "news", "benchmark"}
    assert isinstance(cleaned["portfolio"], list)
    assert {p["ticker"] for p in cleaned["portfolio"]} == {"AAPL", "MSFT"}

    assert set(cleaned["prices"].keys()) == {"AAPL", "MSFT"}
    for ticker, df in cleaned["prices"].items():
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
        assert len(df) >= 252

    for ticker in ("AAPL", "MSFT"):
        assert ticker in cleaned["news"]
        assert len(cleaned["news"][ticker]) <= 10

    assert isinstance(cleaned["benchmark"], pd.DataFrame)
    assert not cleaned["benchmark"].empty
    assert list(cleaned["benchmark"].columns) == ["date", "open", "high", "low", "close", "volume"]

    # Smoke check against spec benchmark ticker usage
    assert BENCHMARK_TICKER == "SPY"
