from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

from config.settings import RISK_FREE_RATE

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class RiskAnalyzer:
    """Portfolio and per-ticker risk metrics (Sharpe, Sortino, VaR, drawdown, beta)."""

    TRADING_DAYS = 252

    def __init__(
        self,
        prices: Dict[str, pd.DataFrame],
        benchmark: pd.DataFrame,
        portfolio: List[Dict[str, Any]],
        risk_free_rate: float = RISK_FREE_RATE,
    ):
        self.prices = prices
        self.benchmark = benchmark
        self.portfolio = portfolio if portfolio is not None else []
        self.risk_free_rate = float(risk_free_rate)

    def _total_market_value(self) -> float:
        total = 0.0
        for pos in self.portfolio:
            mv = pos.get("market_value")
            if mv is not None:
                try:
                    total += float(mv)
                except (TypeError, ValueError):
                    continue
        return total

    def _market_value_for_ticker(self, ticker: str) -> float:
        t = str(ticker).upper().strip()
        for pos in self.portfolio:
            if str(pos.get("ticker", "")).upper().strip() == t:
                mv = pos.get("market_value")
                if mv is not None:
                    try:
                        return float(mv)
                    except (TypeError, ValueError):
                        return 0.0
        return 0.0

    def _prep_price_df(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        out = df.copy()
        if "date" not in out.columns:
            logger.warning("RiskAnalyzer: price data missing 'date' column")
            return None
        if "close" not in out.columns:
            logger.warning("RiskAnalyzer: price data missing 'close' column")
            return None
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date", "close"]).sort_values("date")
        return out

    def _get_returns(self, ticker: str) -> pd.Series:
        if ticker not in self.prices:
            logger.warning("RiskAnalyzer._get_returns: unknown ticker %r", ticker)
            return pd.Series(dtype=float)
        df = self._prep_price_df(self.prices[ticker])
        if df is None or df.empty:
            return pd.Series(dtype=float)
        s = df.set_index("date")["close"].astype(float).pct_change().dropna()
        s.name = "return"
        return s

    def _get_benchmark_returns(self) -> pd.Series:
        df = self._prep_price_df(self.benchmark)
        if df is None or df.empty:
            logger.warning("RiskAnalyzer: benchmark series empty; cannot align")
            return pd.Series(dtype=float)
        s = df.set_index("date")["close"].astype(float).pct_change().dropna()
        s.name = "spy_return"
        return s

    def _get_portfolio_returns(self) -> pd.Series:
        total_mv = self._total_market_value()
        if total_mv <= 0:
            logger.warning(
                "RiskAnalyzer._get_portfolio_returns: zero or missing total market value"
            )
            return pd.Series(dtype=float)

        weighted: List[tuple[str, float]] = []
        return_series: Dict[str, pd.Series] = {}
        for pos in self.portfolio:
            sym = pos.get("ticker")
            mv = pos.get("market_value")
            if sym is None or mv is None:
                continue
            try:
                w = float(mv) / total_mv
            except (TypeError, ValueError):
                continue
            if w <= 0:
                continue
            t = str(sym).upper().strip()
            df = self._prep_price_df(self.prices.get(t))
            if df is None or df.empty:
                logger.warning(
                    "RiskAnalyzer._get_portfolio_returns: no returns for %r; skipping weight",
                    t,
                )
                continue
            series = df.copy()
            series["date"] = pd.to_datetime(series["date"]).dt.normalize()
            series = series.set_index("date")["close"].astype(float)
            returns = series.pct_change().dropna()
            returns.name = t
            if returns.empty:
                logger.warning(
                    "RiskAnalyzer._get_portfolio_returns: no returns for %r; skipping weight",
                    t,
                )
                continue
            logger.debug(
                "RiskAnalyzer._get_portfolio_returns: %s returns length=%d range=%s -> %s",
                t,
                len(returns),
                returns.index.min(),
                returns.index.max(),
            )
            weighted.append((t, w))
            return_series[t] = returns

        if not weighted or not return_series:
            logger.warning("RiskAnalyzer._get_portfolio_returns: no usable ticker returns")
            return pd.Series(dtype=float)

        combined = pd.concat(
            [return_series[ticker] for ticker, _ in weighted],
            axis=1,
            join="inner",
        )
        if combined.empty:
            logger.warning(
                "RiskAnalyzer._get_portfolio_returns: empty intersection of return dates"
            )
            return pd.Series(dtype=float)

        weights = pd.Series({ticker: weight for ticker, weight in weighted}, dtype=float)
        out = combined.mul(weights, axis=1).sum(axis=1)
        out.name = "portfolio_return"
        return out.sort_index()

    def _returns_for_scope(self, ticker: Optional[str]) -> pd.Series:
        if ticker is None:
            return self._get_portfolio_returns()
        return self._get_returns(ticker)

    @staticmethod
    def _annualized_return(daily: pd.Series) -> float:
        if daily.empty:
            return float("nan")
        return float(daily.mean() * RiskAnalyzer.TRADING_DAYS)

    @staticmethod
    def _annualized_vol(daily: pd.Series) -> float:
        if daily.empty or len(daily) < 2:
            return float("nan")
        return float(daily.std(ddof=1) * np.sqrt(RiskAnalyzer.TRADING_DAYS))

    def sharpe_ratio(self, ticker: Optional[str] = None) -> Optional[Dict[str, Any]]:
        label = ticker if ticker is not None else "portfolio"
        logger.info("RiskAnalyzer.sharpe_ratio: %s", label)
        r = self._returns_for_scope(ticker)
        if r.empty or len(r) < 2:
            logger.warning("RiskAnalyzer.sharpe_ratio: insufficient returns for %s", label)
            return None

        ann_ret = self._annualized_return(r)
        ann_vol = self._annualized_vol(r)
        if ann_vol == 0 or np.isnan(ann_vol):
            logger.warning(
                "RiskAnalyzer.sharpe_ratio: zero or NaN volatility for %s", label
            )
            return {
                "ticker": label,
                "sharpe": float("nan"),
                "annualized_return": ann_ret,
                "annualized_volatility": ann_vol,
            }

        sharpe = (ann_ret - self.risk_free_rate) / ann_vol
        return {
            "ticker": label,
            "sharpe": float(sharpe),
            "annualized_return": float(ann_ret),
            "annualized_volatility": float(ann_vol),
        }

    @staticmethod
    def _downside_deviation_daily(daily: pd.Series, mar: float = 0.0) -> float:
        if daily.empty:
            return float("nan")
        downside = np.minimum(0.0, daily.values.astype(float) - mar)
        return float(np.sqrt(np.mean(downside**2)))

    def sortino_ratio(self, ticker: Optional[str] = None) -> Optional[Dict[str, Any]]:
        label = ticker if ticker is not None else "portfolio"
        logger.info("RiskAnalyzer.sortino_ratio: %s", label)
        r = self._returns_for_scope(ticker)
        if r.empty or len(r) < 2:
            logger.warning("RiskAnalyzer.sortino_ratio: insufficient returns for %s", label)
            return None

        ann_ret = self._annualized_return(r)
        ann_vol = self._annualized_vol(r)
        ddown = self._downside_deviation_daily(r, mar=0.0)
        ann_down = ddown * np.sqrt(self.TRADING_DAYS)

        if ann_down == 0 or np.isnan(ann_down):
            logger.warning(
                "RiskAnalyzer.sortino_ratio: zero downside deviation for %s", label
            )
            sortino = float("nan")
        else:
            sortino = (ann_ret - self.risk_free_rate) / ann_down

        return {
            "ticker": label,
            "sortino": float(sortino),
            "annualized_return": float(ann_ret),
            "annualized_volatility": float(ann_vol),
        }

    def max_drawdown(self, ticker: Optional[str] = None) -> Optional[Dict[str, Any]]:
        label = ticker if ticker is not None else "portfolio"
        logger.info("RiskAnalyzer.max_drawdown: %s", label)
        r = self._returns_for_scope(ticker)
        if r.empty:
            logger.warning("RiskAnalyzer.max_drawdown: no returns for %s", label)
            return None

        # Equity curve (1 + r).cumprod(); drawdown matches peak-to-trough on this curve.
        cumulative_returns = (1.0 + r).cumprod()
        rolling_max = cumulative_returns.cummax()
        rm = rolling_max.replace(0, np.nan)
        drawdown = (cumulative_returns - rolling_max) / rm
        drawdown = drawdown.fillna(0.0)
        max_dd = float(drawdown.min())
        pct = f"{max_dd * 100:.1f}%"
        return {
            "ticker": label,
            "max_drawdown": max_dd,
            "max_drawdown_pct": pct,
            "drawdown_series": drawdown,
        }

    def var_95(self, ticker: Optional[str] = None) -> Optional[Dict[str, Any]]:
        label = ticker if ticker is not None else "portfolio"
        logger.info("RiskAnalyzer.var_95: %s", label)
        r = self._returns_for_scope(ticker)
        if r.empty:
            logger.warning("RiskAnalyzer.var_95: no returns for %s", label)
            return None

        var_95 = float(np.percentile(r.values, 5))
        tail = r[r <= var_95]
        cvar_95 = float(tail.mean()) if len(tail) else float("nan")

        if ticker is None:
            pv = self._total_market_value()
        else:
            pv = self._market_value_for_ticker(ticker)
        var_95_dollar = float(var_95 * pv)

        return {
            "ticker": label,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "var_95_dollar": var_95_dollar,
        }

    def beta_alpha(self, ticker: Optional[str] = None) -> Optional[Dict[str, Any]]:
        label = ticker if ticker is not None else "portfolio"
        logger.info("RiskAnalyzer.beta_alpha: %s vs benchmark", label)
        asset_r = self._returns_for_scope(ticker)
        bench_r = self._get_benchmark_returns()
        if asset_r.empty or bench_r.empty:
            logger.warning("RiskAnalyzer.beta_alpha: missing returns for %s", label)
            return None

        aligned = pd.concat([asset_r, bench_r], axis=1, join="inner").dropna()
        if len(aligned) < 2:
            logger.warning(
                "RiskAnalyzer.beta_alpha: insufficient overlapping dates for %s", label
            )
            return None

        y = aligned.iloc[:, 0].astype(float)
        x = aligned.iloc[:, 1].astype(float)
        var_b = float(x.var(ddof=1))
        if var_b == 0 or np.isnan(var_b):
            logger.warning("RiskAnalyzer.beta_alpha: zero benchmark variance for %s", label)
            return {
                "ticker": label,
                "beta": float("nan"),
                "alpha": float("nan"),
                "r_squared": float("nan"),
                "correlation": float("nan"),
            }

        cov = float(y.cov(x))
        beta = cov / var_b
        alpha = float((y - beta * x).mean() * self.TRADING_DAYS - self.risk_free_rate)

        corr = float(y.corr(x)) if len(y) > 1 else float("nan")
        try:
            _, _, r_val, _, _ = linregress(x.values, y.values)
            r_squared = float(r_val**2)
        except Exception:
            r_squared = float("nan")

        return {
            "ticker": label,
            "beta": float(beta),
            "alpha": float(alpha),
            "r_squared": r_squared,
            "correlation": corr,
        }

    def run_all(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        logger.info("RiskAnalyzer.run_all: portfolio-level metrics")
        out["portfolio"] = {
            "sharpe": self.sharpe_ratio(None),
            "sortino": self.sortino_ratio(None),
            "max_drawdown": self.max_drawdown(None),
            "var_95": self.var_95(None),
            "beta_alpha": self.beta_alpha(None),
        }

        for ticker in self.prices:
            logger.info("RiskAnalyzer.run_all: ticker %r", ticker)
            out[ticker] = {
                "sharpe": self.sharpe_ratio(ticker),
                "sortino": self.sortino_ratio(ticker),
                "max_drawdown": self.max_drawdown(ticker),
                "var_95": self.var_95(ticker),
                "beta_alpha": self.beta_alpha(ticker),
            }

        return out
