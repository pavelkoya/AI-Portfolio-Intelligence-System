from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

import pandas as pd
import pandas_ta  # noqa: F401 — registers `df.ta` accessor

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class TechnicalAnalyzer:
    """Technical indicators (RSI, MACD, moving averages) via pandas_ta."""

    MIN_ROWS_MA = 200
    MIN_ROWS_MACD = 40
    MIN_ROWS_RSI_EXTRA = 5

    def __init__(self, prices: Dict[str, pd.DataFrame]):
        self.prices = prices

    def _get_work_frame(self, ticker: str) -> Optional[pd.DataFrame]:
        if ticker not in self.prices:
            logger.warning("TechnicalAnalyzer: ticker %r not in prices", ticker)
            return None
        df = self.prices[ticker]
        if df is None or df.empty:
            logger.warning("TechnicalAnalyzer: empty dataframe for %r", ticker)
            return None
        out = df.copy()
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out = out.dropna(subset=["date"]).sort_values("date")
        if "close" not in out.columns:
            logger.warning("TechnicalAnalyzer: missing 'close' column for %r", ticker)
            return None
        return out.reset_index(drop=True)

    @staticmethod
    def _macd_column_names(mdf: pd.DataFrame) -> tuple[str, str, str]:
        cols = list(mdf.columns)
        line_col = next(
            (
                c
                for c in cols
                if re.fullmatch(r"MACD_\d+_\d+_\d+", c) is not None
            ),
            None,
        )
        signal_col = next(
            (
                c
                for c in cols
                if c.startswith("MACDs_") or c.startswith("MACDS_")
            ),
            None,
        )
        # Histogram: pandas-ta uses MACDh_ (freqtrade fork) or MACDH_ in some builds
        hist_col = next(
            (c for c in cols if c.startswith("MACDh_") or c.startswith("MACDH_")),
            None,
        )
        if not all([line_col, signal_col, hist_col]):
            raise ValueError(f"Unexpected MACD columns: {cols}")
        return line_col, signal_col, hist_col

    @staticmethod
    def _macd_crossover_last_n_bars(
        macd_line: pd.Series, signal_line: pd.Series, n: int = 3
    ) -> bool:
        """True if MACD crossed the signal line in any of the last `n` bar transitions."""
        aligned = pd.DataFrame({"macd": macd_line, "signal": signal_line}).dropna()
        if len(aligned) < 2:
            return False
        window = max(2, min(n + 1, len(aligned)))
        sub = aligned.iloc[-window:]
        for i in range(1, len(sub)):
            prev_above = bool(sub.iloc[i - 1]["macd"] > sub.iloc[i - 1]["signal"])
            curr_above = bool(sub.iloc[i]["macd"] > sub.iloc[i]["signal"])
            if prev_above != curr_above:
                return True
        return False

    def calculate_rsi(self, ticker: str, period: int = 14) -> Optional[Dict[str, Any]]:
        df = self._get_work_frame(ticker)
        if df is None:
            return None
        need = period + self.MIN_ROWS_RSI_EXTRA
        if len(df) < need:
            logger.warning(
                "TechnicalAnalyzer.calculate_rsi: insufficient rows for %r "
                "(%d < %d)",
                ticker,
                len(df),
                need,
            )
            return None
        logger.info("TechnicalAnalyzer.calculate_rsi: %r period=%d", ticker, period)
        rsi_series = df.ta.rsi(close=df["close"], length=period)
        if rsi_series is None or rsi_series.dropna().empty:
            logger.warning("TechnicalAnalyzer.calculate_rsi: no RSI values for %r", ticker)
            return None
        last = float(rsi_series.dropna().iloc[-1])
        if last > 70:
            signal = "Overbought"
        elif last < 30:
            signal = "Oversold"
        else:
            signal = "Neutral"
        return {
            "ticker": ticker,
            "rsi_current": last,
            "signal": signal,
            "values": rsi_series,
        }

    def calculate_macd(self, ticker: str) -> Optional[Dict[str, Any]]:
        df = self._get_work_frame(ticker)
        if df is None:
            return None
        if len(df) < self.MIN_ROWS_MACD:
            logger.warning(
                "TechnicalAnalyzer.calculate_macd: insufficient rows for %r "
                "(%d < %d)",
                ticker,
                len(df),
                self.MIN_ROWS_MACD,
            )
            return None
        logger.info("TechnicalAnalyzer.calculate_macd: %r", ticker)
        macd_df = df.ta.macd(close=df["close"], fast=12, slow=26, signal=9)
        if macd_df is None or macd_df.empty:
            logger.warning("TechnicalAnalyzer.calculate_macd: empty result for %r", ticker)
            return None
        line_c, signal_c, hist_c = self._macd_column_names(macd_df)
        macd_line_s = macd_df[line_c]
        signal_line_s = macd_df[signal_c]
        hist_s = macd_df[hist_c]
        valid = macd_df[[line_c, signal_c, hist_c]].dropna()
        if valid.empty:
            logger.warning(
                "TechnicalAnalyzer.calculate_macd: no valid MACD rows for %r", ticker
            )
            return None
        last_row = valid.iloc[-1]
        macd_v = float(last_row[line_c])
        sig_v = float(last_row[signal_c])
        hist_v = float(last_row[hist_c])
        direction = "Bullish" if macd_v > sig_v else "Bearish"
        crossover = self._macd_crossover_last_n_bars(macd_line_s, signal_line_s, n=3)
        return {
            "ticker": ticker,
            "macd_line": macd_v,
            "signal_line": sig_v,
            "histogram": hist_v,
            "signal": direction,
            "crossover": crossover,
        }

    def calculate_ma(self, ticker: str) -> Optional[Dict[str, Any]]:
        df = self._get_work_frame(ticker)
        if df is None:
            return None
        if len(df) < self.MIN_ROWS_MA:
            logger.warning(
                "TechnicalAnalyzer.calculate_ma: insufficient rows for %r "
                "(%d < %d for SMA 200)",
                ticker,
                len(df),
                self.MIN_ROWS_MA,
            )
            return None
        logger.info("TechnicalAnalyzer.calculate_ma: %r", ticker)
        sma50 = df.ta.sma(close=df["close"], length=50)
        sma200 = df.ta.sma(close=df["close"], length=200)
        if sma50 is None or sma200 is None:
            return None
        combined = pd.DataFrame(
            {"close": df["close"], "sma_50": sma50, "sma_200": sma200}
        ).dropna()
        if combined.empty:
            logger.warning("TechnicalAnalyzer.calculate_ma: no valid MA rows for %r", ticker)
            return None
        last = combined.iloc[-1]
        close_v = float(last["close"])
        s50 = float(last["sma_50"])
        s200 = float(last["sma_200"])
        return {
            "ticker": ticker,
            "sma_50": s50,
            "sma_200": s200,
            "current_price": close_v,
            "above_sma_50": close_v > s50,
            "above_sma_200": close_v > s200,
            "golden_cross": s50 > s200,
            "death_cross": s50 < s200,
        }

    def run_all(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for ticker in self.prices:
            df = self._get_work_frame(ticker)
            if df is None or len(df) < self.MIN_ROWS_MA:
                logger.warning(
                    "TechnicalAnalyzer.run_all: skipping %r (need >= %d rows)",
                    ticker,
                    self.MIN_ROWS_MA,
                )
                out[ticker] = None
                continue
            logger.info("TechnicalAnalyzer.run_all: processing %r", ticker)
            out[ticker] = {
                "rsi": self.calculate_rsi(ticker),
                "macd": self.calculate_macd(ticker),
                "ma": self.calculate_ma(ticker),
            }
        return out
