from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.settings import TRADING_DAYS


class RiskLevelCalculator:
    def __init__(
        self,
        prices: dict[str, pd.DataFrame],
        portfolio: list[dict],
        garch_output: dict = None,
    ):
        self.prices = prices
        self.portfolio = portfolio
        self.garch_output = garch_output or {}
        self.logger = logging.getLogger(__name__)
        self.results: dict[str, dict] = {}

    def _get_current_price(self, ticker: str) -> float:
        t = str(ticker).upper().strip()
        for pos in self.portfolio or []:
            if str(pos.get("ticker", "")).upper().strip() == t:
                cp = pos.get("current_price")
                if cp is None:
                    break
                try:
                    return float(cp)
                except (TypeError, ValueError):
                    break

        df = (self.prices or {}).get(t)
        if df is None or getattr(df, "empty", True) or "close" not in df.columns:
            return None
        try:
            tmp = df.copy()
            if "date" in tmp.columns:
                tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
                tmp = tmp.dropna(subset=["date"]).sort_values("date")
            close = pd.to_numeric(tmp["close"], errors="coerce").dropna()
            if close.empty:
                return None
            return float(close.iloc[-1])
        except Exception:
            return None

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        if df is None or df.empty:
            return None
        if len(df) < period + 1:
            return None
        required = {"high", "low", "close"}
        if not required.issubset(set(df.columns)):
            return None

        tmp = df.copy()
        tmp["high"] = pd.to_numeric(tmp["high"], errors="coerce")
        tmp["low"] = pd.to_numeric(tmp["low"], errors="coerce")
        tmp["close"] = pd.to_numeric(tmp["close"], errors="coerce")
        tmp = tmp.dropna(subset=["high", "low", "close"])
        if len(tmp) < period + 1:
            return None

        tmp["prev_close"] = tmp["close"].shift(1)
        tmp["tr1"] = tmp["high"] - tmp["low"]
        tmp["tr2"] = (tmp["high"] - tmp["prev_close"]).abs()
        tmp["tr3"] = (tmp["low"] - tmp["prev_close"]).abs()
        tmp["tr"] = tmp[["tr1", "tr2", "tr3"]].max(axis=1)

        atr_series = tmp["tr"].ewm(alpha=1 / period, adjust=False).mean()
        if atr_series.empty or pd.isna(atr_series.iloc[-1]):
            return None
        return float(atr_series.iloc[-1])

    def _get_atr_multipliers(self, ticker: str) -> tuple:
        t = str(ticker).upper().strip()
        regime = (self.garch_output.get(t) or {}).get("vol_regime", "Low")
        if regime == "Stress":
            stop_mult = 2.5
            target_mult = 3.75  # maintains 1.5:1 R:R
        else:
            stop_mult = 2.0
            target_mult = 3.0  # standard 1.5:1 R:R
        return (stop_mult, target_mult)

    def _empty_result(self, ticker: str) -> dict:
        t = str(ticker).upper().strip()
        vol_regime = (self.garch_output.get(t) or {}).get("vol_regime", "Unknown")
        return {
            "ticker": t,
            "current_price": None,
            "atr_14": None,
            "stop_loss": None,
            "take_profit": None,
            "stop_distance_pct": None,
            "target_distance_pct": None,
            "risk_reward": None,
            "stop_multiplier": None,
            "vol_regime": vol_regime,
        }

    def calculate_ticker(self, ticker: str) -> dict:
        t = str(ticker).upper().strip()
        try:
            df = (self.prices or {}).get(t)
            if df is None or getattr(df, "empty", True):
                self.logger.warning("RiskLevelCalculator: missing/empty prices for %s", t)
                return self._empty_result(t)

            tmp = df.copy()
            if "date" in tmp.columns:
                tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.normalize()
                tmp = tmp.dropna(subset=["date"]).sort_values("date")
            tmp = tmp.tail(30)

            atr = self._compute_atr(tmp, period=14)
            if atr is None:
                self.logger.warning("RiskLevelCalculator: ATR unavailable for %s", t)
                return self._empty_result(t)

            current_price = self._get_current_price(t)
            if current_price is None or current_price <= 0:
                self.logger.warning("RiskLevelCalculator: current_price unavailable for %s", t)
                return self._empty_result(t)

            stop_mult, target_mult = self._get_atr_multipliers(t)
            stop_loss = float(current_price) - (float(atr) * float(stop_mult))
            take_profit = float(current_price) + (float(atr) * float(target_mult))

            try:
                assert stop_loss < current_price
                assert take_profit > current_price
            except AssertionError:
                self.logger.critical(
                    "RiskLevelCalculator: invalid levels for %s (price=%s stop=%s target=%s)",
                    t,
                    current_price,
                    stop_loss,
                    take_profit,
                )
                return self._empty_result(t)

            risk_reward = (take_profit - current_price) / (current_price - stop_loss)
            if abs(float(risk_reward) - 1.5) > 0.05:
                self.logger.warning(
                    "RiskLevelCalculator: R:R deviates for %s (rr=%.3f)",
                    t,
                    float(risk_reward),
                )

            vol_regime = (self.garch_output.get(t) or {}).get("vol_regime", "Unknown")
            out = {
                "ticker": t,
                "current_price": round(float(current_price), 2),
                "atr_14": round(float(atr), 4),
                "stop_loss": round(float(stop_loss), 2),
                "take_profit": round(float(take_profit), 2),
                "stop_distance_pct": round((current_price - stop_loss) / current_price, 4),
                "target_distance_pct": round((take_profit - current_price) / current_price, 4),
                "risk_reward": round(float(risk_reward), 2),
                "stop_multiplier": float(stop_mult),
                "vol_regime": vol_regime,
            }
            return out
        except Exception:
            self.logger.exception("RiskLevelCalculator.calculate_ticker failed for %s", t)
            return self._empty_result(t)

    def run_all(self) -> dict:
        for ticker in sorted((self.prices or {}).keys()):
            res = self.calculate_ticker(ticker)
            self.results[str(ticker).upper().strip()] = res

        for ticker in sorted(self.results.keys()):
            r = self.results.get(ticker) or {}
            self.logger.info(
                "%s: price=%s stop=%s target=%s regime=%s",
                ticker,
                r.get("current_price"),
                r.get("stop_loss"),
                r.get("take_profit"),
                r.get("vol_regime"),
            )
        return self.results

    def get_committee_input(self) -> dict:
        if not self.results:
            self.run_all()
        return self.results

    def get_summary_df(self) -> pd.DataFrame:
        if not self.results:
            self.run_all()

        df = pd.DataFrame.from_dict(self.results, orient="index")
        if df.empty:
            return df

        cols = [
            "ticker",
            "current_price",
            "atr_14",
            "stop_loss",
            "take_profit",
            "stop_distance_pct",
            "target_distance_pct",
            "risk_reward",
            "vol_regime",
        ]
        df = df.reindex(columns=cols)

        for c in [
            "current_price",
            "atr_14",
            "stop_loss",
            "take_profit",
            "stop_distance_pct",
            "target_distance_pct",
            "risk_reward",
        ]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

        if "stop_distance_pct" in df.columns:
            df = df.sort_values("stop_distance_pct", ascending=True, na_position="last")
        return df.reset_index(drop=True)

