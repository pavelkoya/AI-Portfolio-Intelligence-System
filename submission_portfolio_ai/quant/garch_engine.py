from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from arch import arch_model

from config.settings import TRADING_DAYS
from data.database import DatabaseManager


class GARCHEngine:
    def __init__(self, prices: dict[str, pd.DataFrame]):
        # prices: dict of {ticker: OHLCV DataFrame} same format as Phase 2 input
        self.prices = prices
        self.results: dict[str, dict] = {}  # stores output per ticker
        self.logger = logging.getLogger(__name__)

    def _prepare_returns(self, ticker: str) -> pd.Series:
        df = self.prices.get(ticker)
        if df is None or getattr(df, "empty", True):
            self.logger.warning("GARCHEngine: missing/empty prices for %s", ticker)
            return None

        if "close" not in df.columns:
            self.logger.warning("GARCHEngine: price data missing 'close' for %s", ticker)
            return None
        if "date" not in df.columns:
            self.logger.warning("GARCHEngine: price data missing 'date' for %s", ticker)
            return None

        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        tmp["close"] = pd.to_numeric(tmp["close"], errors="coerce")
        tmp = tmp.dropna(subset=["date", "close"]).sort_values("date")
        tmp = tmp.set_index("date")

        close = tmp["close"]
        returns = np.log(close / close.shift(1)).dropna() * 100.0  # percent returns

        if len(returns) < 252:
            self.logger.warning(
                "GARCHEngine: insufficient history for %s (%d rows; need >= 252)",
                ticker,
                len(returns),
            )
            return None

        returns.name = "returns"
        return returns

    def _vol_regime_label(
        self,
        current_vol: float,
        historical_vols: pd.Series,
    ) -> str:
        hist = pd.to_numeric(historical_vols, errors="coerce").dropna()
        if hist.empty:
            return "Unknown"
        p33 = np.percentile(hist, 33)
        p66 = np.percentile(hist, 66)
        if current_vol <= p33:
            return "Low"
        if current_vol <= p66:
            return "Elevated"
        return "Stress"

    def fit_ticker(self, ticker: str) -> dict:
        self.logger.info("GARCHEngine.fit_ticker: starting %s", ticker)
        try:
            returns = self._prepare_returns(ticker)
            if returns is None:
                self.logger.warning("GARCHEngine.fit_ticker: skipped %s (no returns)", ticker)
                return None

            model = arch_model(
                returns,
                vol="Garch",
                p=1,
                q=1,
                dist="normal",
                rescale=False,
            )
            result = model.fit(disp="off")

            forecast = result.forecast(horizon=10)
            variance_forecast = forecast.variance.iloc[-1]
            vol_forecast_10d = np.sqrt(variance_forecast.values)

            last_vol_daily = np.sqrt(result.conditional_volatility.iloc[-1])
            annualized_vol = last_vol_daily * np.sqrt(TRADING_DAYS) / 100.0

            hist_vols = result.conditional_volatility / 100.0
            regime = self._vol_regime_label(last_vol_daily / 100.0, hist_vols.tail(252))
            self.logger.info("GARCHEngine.fit_ticker: %s regime=%s", ticker, regime)

            out = {
                "ticker": ticker,
                "forecast_vol_10d": vol_forecast_10d.tolist(),
                "annualized_vol": round(float(annualized_vol), 4),
                "vol_regime": regime,
                "last_conditional_vol": round(float(last_vol_daily / 100.0), 4),
            }
            self.logger.info("GARCHEngine.fit_ticker: finished %s", ticker)
            return out
        except Exception:
            self.logger.exception("GARCHEngine.fit_ticker: failed for %s", ticker)
            return {
                "ticker": ticker,
                "forecast_vol_10d": None,
                "annualized_vol": None,
                "vol_regime": "Unknown",
                "last_conditional_vol": None,
            }

    def run_all(self) -> dict:
        regime_counts: dict[str, int] = {}
        n = 0
        for ticker in sorted(self.prices.keys()):
            res = self.fit_ticker(ticker)
            self.results[ticker] = res
            if res is not None:
                n += 1
                regime = res.get("vol_regime", "Unknown")
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            else:
                self.logger.warning("GARCHEngine.run_all: no result for %s", ticker)

        self.logger.info("GARCH complete: %d tickers, regimes: %s", n, regime_counts)
        return self.results

    def get_committee_input(self) -> dict:
        if not self.results:
            self.run_all()

        out: dict[str, dict] = {}
        for ticker, res in self.results.items():
            if not res:
                continue
            out[ticker] = {
                "annualized_vol": res.get("annualized_vol"),
                "vol_regime": res.get("vol_regime"),
                "last_conditional_vol": res.get("last_conditional_vol"),
            }
        return out

