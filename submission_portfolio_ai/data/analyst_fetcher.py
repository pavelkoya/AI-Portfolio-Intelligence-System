from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf


class AnalystFetcher:
    def __init__(self, portfolio: list[dict]):
        self.portfolio = portfolio
        self.logger = logging.getLogger(__name__)
        self.results: dict[str, dict] = {}

    def _get_current_price(self, ticker: str) -> float:
        t = str(ticker).upper().strip()
        for pos in self.portfolio or []:
            if str(pos.get("ticker", "")).upper().strip() == t:
                cp = pos.get("current_price")
                if cp is None:
                    return None
                try:
                    return float(cp)
                except (TypeError, ValueError):
                    return None
        return None

    def _compute_signal(self, upside_pct: float) -> str:
        if upside_pct > 0.10:
            return "BUY"
        if upside_pct < -0.05:
            return "REDUCE"
        return "HOLD"

    def fetch_ticker(self, ticker: str) -> dict:
        try:
            info = yf.Ticker(ticker).info

            current_price = self._get_current_price(ticker)
            if current_price is None:
                current_price = info.get("currentPrice") or info.get("regularMarketPrice")

            target_price = info.get("targetMeanPrice")
            analyst_count = info.get("numberOfAnalystOpinions", 0)
            recommendation = info.get("recommendationKey", "") or ""

            rec_map = {
                "strong_buy": "BUY",
                "buy": "BUY",
                "hold": "HOLD",
                "sell": "REDUCE",
                "strong_sell": "REDUCE",
                "underperform": "REDUCE",
                "outperform": "BUY",
                "neutral": "HOLD",
                "overweight": "BUY",
                "underweight": "REDUCE",
            }

            if target_price and current_price and float(current_price) > 0:
                upside_pct = (float(target_price) - float(current_price)) / float(
                    current_price
                )
                upside_pct = round(float(upside_pct), 4)
                signal = self._compute_signal(upside_pct)
            elif recommendation:
                upside_pct = None
                signal = rec_map.get(recommendation.lower(), "HOLD")
            else:
                upside_pct = None
                signal = "NO DATA"

            return {
                "ticker": str(ticker).upper().strip(),
                "target_price": round(float(target_price), 2) if target_price else None,
                "current_price": float(current_price) if current_price else None,
                "analyst_count": int(analyst_count or 0),
                "most_recent_date": None,
                "upside_pct": upside_pct,
                "upside_pct_str": f"{upside_pct * 100:.1f}%" if upside_pct else "N/A",
                "signal": signal,
                "recommendation": recommendation or "N/A",
            }
        except Exception as e:
            self.logger.error("fetch_ticker %s failed: %s", ticker, e)
            return {
                "ticker": str(ticker).upper().strip(),
                "target_price": None,
                "analyst_count": 0,
                "upside_pct": None,
                "upside_pct_str": "N/A",
                "signal": "ERROR",
                "recommendation": "N/A",
            }

    def run_all(self) -> dict:
        tickers = sorted(
            {
                str(p.get("ticker", "")).upper().strip()
                for p in (self.portfolio or [])
                if str(p.get("ticker", "")).strip()
            }
        )

        counts = {"BUY": 0, "HOLD": 0, "REDUCE": 0, "NO DATA": 0, "ERROR": 0}
        for t in tickers:
            res = self.fetch_ticker(t)
            self.results[t] = res
            sig = (res or {}).get("signal")
            if sig in counts:
                counts[sig] += 1

        self.logger.info("Analyst targets fetched for %d tickers (source=yfinance)", len(tickers))
        self.logger.info(
            "BUY: %d, HOLD: %d, REDUCE: %d, NO DATA: %d",
            counts["BUY"],
            counts["HOLD"],
            counts["REDUCE"],
            counts["NO DATA"],
        )
        return self.results

    def get_committee_input(self) -> dict:
        if not self.results:
            self.run_all()
        return self.results

    def get_summary_df(self) -> pd.DataFrame:
        if not self.results:
            self.run_all()

        records: list[dict] = []
        for ticker in sorted(self.results.keys()):
            res = self.results.get(ticker) or {}
            records.append(
                {
                    "ticker": ticker,
                    "target_price": res.get("target_price"),
                    "current_price": res.get("current_price", self._get_current_price(ticker)),
                    "upside_pct_str": res.get("upside_pct_str"),
                    "analyst_count": res.get("analyst_count"),
                    "signal": res.get("signal"),
                    "most_recent_date": res.get("most_recent_date"),
                }
            )

        df = pd.DataFrame.from_records(
            records,
            columns=[
                "ticker",
                "target_price",
                "current_price",
                "upside_pct_str",
                "analyst_count",
                "signal",
                "most_recent_date",
            ],
        )

        def _upside_sort_key(v: Any) -> float:
            if v is None:
                return float("-inf")
            try:
                s = str(v).strip()
                if s.endswith("%"):
                    s = s[:-1]
                return float(s)
            except Exception:
                return float("-inf")

        if not df.empty:
            df["_upside_sort"] = df["upside_pct_str"].apply(_upside_sort_key)
            df = df.sort_values("_upside_sort", ascending=False).drop(columns=["_upside_sort"])
        return df
