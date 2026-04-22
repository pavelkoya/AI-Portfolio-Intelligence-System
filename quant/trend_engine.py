from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import linregress

from config.settings import TRADING_DAYS


class TrendEngine:
    def __init__(self, prices: dict[str, pd.DataFrame]):
        self.prices = prices
        self.results: dict[str, dict] = {}
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _empty_result(ticker: str) -> dict:
        return {
            "ticker": ticker,
            "trend_direction": None,
            "trend_slope_normalized": None,
            "trend_slope_acceleration": None,
            "trend_deviation_std": None,
            "trend_uncertainty_pct": None,
            "seasonal_component_pct": None,
            "trend_confidence_score": None,
            "window_days": None,
            "current_price": None,
        }

    def _compute_trend(self, ticker: str, closes: pd.Series) -> dict:
        if closes is None or closes.empty:
            self.logger.warning("TrendEngine._compute_trend: %s has no close data", ticker)
            return None

        closes = pd.to_numeric(closes, errors="coerce").dropna()
        if len(closes) < 30:
            self.logger.warning(
                "TrendEngine._compute_trend: %s has insufficient rows (%d; need >=30)",
                ticker,
                len(closes),
            )
            return None

        window_days = 90
        if len(closes) < window_days:
            self.logger.warning(
                "TrendEngine._compute_trend: %s has only %d rows; using all available",
                ticker,
                len(closes),
            )
            window = closes.copy()
        else:
            window = closes.tail(window_days).copy()

        x = np.arange(len(window), dtype=float)
        y = window.values.astype(float)
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        trend_direction = "Up" if slope > 0 else "Down"
        mean_price = float(window.mean()) if float(window.mean()) != 0 else np.nan
        trend_slope_normalized = float(slope / mean_price) if np.isfinite(mean_price) else 0.0

        # Acceleration: slope(last 30d) - slope(prior 30d), normalized by mean price.
        if len(window) >= 60:
            recent_30 = window.iloc[-30:]
            prior_30 = window.iloc[-60:-30]
            sx_recent, _, _, _, _ = linregress(
                np.arange(len(recent_30), dtype=float),
                recent_30.values.astype(float),
            )
            sx_prior, _, _, _, _ = linregress(
                np.arange(len(prior_30), dtype=float),
                prior_30.values.astype(float),
            )
            slope_accel = (sx_recent - sx_prior) / mean_price if np.isfinite(mean_price) else 0.0
        else:
            slope_accel = 0.0

        fitted = intercept + slope * x
        residuals = y - fitted
        trend_deviation_std = float(np.std(residuals))

        current_price = float(closes.iloc[-1])
        trend_uncertainty_pct = (
            float(trend_deviation_std / current_price) if current_price > 0 else 0.0
        )

        # Seasonal component: use canonical 30d/90d if available, otherwise scale to available.
        if len(closes) >= 90 and len(closes) >= 30:
            return_30 = (float(closes.iloc[-1]) - float(closes.iloc[-30])) / float(
                closes.iloc[-30]
            )
            return_90 = (float(closes.iloc[-1]) - float(closes.iloc[-90])) / float(
                closes.iloc[-90]
            )
        else:
            lookback = len(closes)
            short_lb = min(30, max(2, lookback // 3))
            base_short = float(closes.iloc[-short_lb]) if short_lb < len(closes) else float(
                closes.iloc[0]
            )
            base_long = float(closes.iloc[0])
            return_30 = (
                (float(closes.iloc[-1]) - base_short) / base_short if base_short != 0 else 0.0
            )
            return_90 = (
                (float(closes.iloc[-1]) - base_long) / base_long if base_long != 0 else 0.0
            )

        if abs(return_90) < 0.001:
            seasonal_component_pct = 0.0
        else:
            seasonal_component_pct = float(return_30 / return_90)

        trend_confidence_score = float(r_value**2)

        return {
            "ticker": ticker,
            "trend_direction": trend_direction,
            "trend_slope_normalized": round(float(trend_slope_normalized), 6),
            "trend_slope_acceleration": round(float(slope_accel), 6),
            "trend_deviation_std": round(float(trend_deviation_std), 4),
            "trend_uncertainty_pct": round(float(trend_uncertainty_pct), 4),
            "seasonal_component_pct": round(float(seasonal_component_pct), 4),
            "trend_confidence_score": round(float(trend_confidence_score), 4),
            "window_days": int(len(window)),
            "current_price": float(current_price),
        }

    def analyze_ticker(self, ticker: str) -> dict:
        t = str(ticker).upper().strip()
        try:
            if t not in self.prices:
                self.logger.warning("TrendEngine.analyze_ticker: unknown ticker %s", t)
                return self._empty_result(t)

            df = self.prices.get(t)
            if df is None or df.empty:
                self.logger.warning("TrendEngine.analyze_ticker: empty dataframe for %s", t)
                return self._empty_result(t)
            if "date" not in df.columns or "close" not in df.columns:
                self.logger.warning(
                    "TrendEngine.analyze_ticker: %s missing required columns", t
                )
                return self._empty_result(t)

            tmp = df.copy()
            tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.normalize()
            tmp["close"] = pd.to_numeric(tmp["close"], errors="coerce")
            tmp = tmp.dropna(subset=["date", "close"]).sort_values("date")
            closes = tmp.set_index("date")["close"]

            trend = self._compute_trend(t, closes)
            if trend is None:
                return self._empty_result(t)

            self.logger.info(
                "%s: dir=%s conf=%.2f slope=%.6f accel=%.6f uncertainty=%.2f%%",
                t,
                trend.get("trend_direction"),
                float(trend.get("trend_confidence_score") or 0.0),
                float(trend.get("trend_slope_normalized") or 0.0),
                float(trend.get("trend_slope_acceleration") or 0.0),
                100.0 * float(trend.get("trend_uncertainty_pct") or 0.0),
            )
            return trend
        except Exception:
            self.logger.exception("TrendEngine.analyze_ticker failed for %s", t)
            return self._empty_result(t)

    def run_all(self) -> dict:
        self.logger.info("TrendEngine.run_all: starting (trading_days=%d)", TRADING_DAYS)
        self.results = {}
        for ticker in sorted((self.prices or {}).keys()):
            t = str(ticker).upper().strip()
            self.results[t] = self.analyze_ticker(t)

        valid = [v for v in self.results.values() if isinstance(v, dict)]
        n = len(self.results)
        n_up = sum(1 for v in valid if v.get("trend_direction") == "Up")
        n_down = sum(1 for v in valid if v.get("trend_direction") == "Down")
        n_high = sum(
            1
            for v in valid
            if v.get("trend_confidence_score") is not None
            and float(v.get("trend_confidence_score")) > 0.6
        )
        n_low = sum(
            1
            for v in valid
            if v.get("trend_confidence_score") is not None
            and float(v.get("trend_confidence_score")) < 0.4
        )

        self.logger.info(
            "Trend complete: %d tickers Up=%d Down=%d High confidence (>0.6): %d Low confidence (<0.4): %d",
            n,
            n_up,
            n_down,
            n_high,
            n_low,
        )
        return self.results

    def get_committee_input(self) -> dict:
        if not self.results:
            self.run_all()
        return {
            ticker: {
                "trend_direction": data.get("trend_direction"),
                "trend_slope_normalized": data.get("trend_slope_normalized"),
                "trend_slope_acceleration": data.get("trend_slope_acceleration"),
                "trend_uncertainty_pct": data.get("trend_uncertainty_pct"),
                "seasonal_component_pct": data.get("seasonal_component_pct"),
                "trend_confidence_score": data.get("trend_confidence_score"),
            }
            for ticker, data in self.results.items()
            if data is not None
        }

    def get_low_confidence_tickers(self, threshold=0.4) -> list:
        if not self.results:
            self.run_all()
        out: list[str] = []
        for ticker, data in self.results.items():
            if not isinstance(data, dict):
                continue
            score = data.get("trend_confidence_score")
            if score is None:
                continue
            try:
                if float(score) < float(threshold):
                    out.append(ticker)
            except (TypeError, ValueError):
                continue
        return out
