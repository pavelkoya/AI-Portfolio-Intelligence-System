from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from config.settings import RISK_FREE_RATE, TRADING_DAYS

logger = logging.getLogger(__name__)


class PostRebalanceEngine:
    def __init__(
        self,
        prices: dict[str, pd.DataFrame],
        benchmark: pd.DataFrame,
        current_portfolio: list[dict],
        proposed_weights: dict,
    ):
        self.prices = prices
        self.benchmark = benchmark
        self.current_portfolio = current_portfolio
        # convert % to decimal and drop CASH
        self.proposed_weights = {
            str(t).upper().strip(): float(w) / 100.0
            for t, w in (proposed_weights or {}).items()
            if str(t).upper().strip() != "CASH"
        }
        self.logger = logging.getLogger(__name__)

    def _get_returns(self, ticker: str) -> pd.Series:
        t = str(ticker).upper().strip()
        df = self.prices.get(t)
        if df is None or df.empty or "date" not in df.columns or "close" not in df.columns:
            return pd.Series(dtype=float)

        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.normalize()
        tmp["close"] = pd.to_numeric(tmp["close"], errors="coerce")
        tmp = tmp.dropna(subset=["date", "close"]).sort_values("date")
        out = tmp.set_index("date")["close"].pct_change().dropna()
        out.name = t
        return out.astype(float)

    def _build_weighted_returns(self, weights: dict) -> pd.Series:
        series_map: dict[str, pd.Series] = {}
        for ticker, w in (weights or {}).items():
            t = str(ticker).upper().strip()
            if t not in self.prices:
                continue
            r = self._get_returns(t)
            if r.empty:
                continue
            series_map[t] = r

        if not series_map:
            return pd.Series(dtype=float)

        combined = pd.concat(series_map, axis=1, join="inner").dropna(how="any")
        if combined.empty:
            return pd.Series(dtype=float)

        weighted = pd.Series(0.0, index=combined.index, dtype=float)
        for t in combined.columns:
            weighted = weighted.add(float(weights.get(t, 0.0)) * combined[t], fill_value=0.0)
        weighted.name = "portfolio_return"
        return weighted

    def _compute_metrics(self, weights: dict, label: str) -> dict:
        weighted_returns = self._build_weighted_returns(weights)
        if weighted_returns.empty:
            self.logger.error("%s metrics: empty weighted returns", label)
            return None

        ann_return = float(weighted_returns.mean() * TRADING_DAYS)
        ann_vol = float(weighted_returns.std() * np.sqrt(TRADING_DAYS))
        if ann_vol == 0:
            sharpe = 0.0
        else:
            sharpe = float((ann_return - RISK_FREE_RATE) / ann_vol)

        cum = (1.0 + weighted_returns).cumprod()
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max
        max_dd = float(drawdown.min())

        var_95 = float(np.percentile(weighted_returns.values, 5))

        # Benchmark returns from self.benchmark DataFrame (not prices["SPY"])
        bench_returns = pd.Series(dtype=float)
        if (
            self.benchmark is not None
            and not self.benchmark.empty
            and "date" in self.benchmark.columns
            and "close" in self.benchmark.columns
        ):
            bench = self.benchmark.copy()
            bench["date"] = pd.to_datetime(bench["date"], errors="coerce").dt.normalize()
            bench["close"] = pd.to_numeric(bench["close"], errors="coerce")
            bench = bench.dropna(subset=["date", "close"]).sort_values("date")
            bench_returns = bench.set_index("date")["close"].pct_change().dropna()

        aligned = pd.concat(
            [weighted_returns.rename("portfolio"), bench_returns.rename("benchmark")],
            axis=1,
            join="inner",
        ).dropna()
        if aligned.empty:
            beta = 1.0
        else:
            cov = float(np.cov(aligned["portfolio"].values, aligned["benchmark"].values)[0, 1])
            var_bench = float(np.var(aligned["benchmark"].values))
            beta = cov / var_bench if var_bench != 0 else 1.0

        # Monte Carlo 5th percentile (as pct of initial value)
        mean_daily = float(weighted_returns.mean())
        std_daily = float(weighted_returns.std())
        initial_value = float(
            sum(float(p.get("market_value", 0.0) or 0.0) for p in (self.current_portfolio or []))
        )
        random_returns = np.random.normal(
            loc=mean_daily,
            scale=std_daily,
            size=(1000, TRADING_DAYS),
        )
        paths = initial_value * np.cumprod(1.0 + random_returns, axis=1)
        final_values = paths[:, -1]
        pct_5th = float(np.percentile(final_values, 5))
        monte_carlo_5th_pct = (
            float((pct_5th - initial_value) / initial_value) if initial_value != 0 else float("nan")
        )

        self.logger.info(
            "%s metrics: sharpe=%.3f dd=%.1f%% var95=%.3f beta=%.3f",
            label,
            sharpe,
            max_dd * 100.0,
            var_95,
            beta,
        )
        return {
            "label": label,
            "sharpe": round(float(sharpe), 3),
            "max_drawdown": round(float(max_dd), 4),
            "var_95": round(float(var_95), 4),
            "beta": round(float(beta), 3),
            "monte_carlo_5th_pct": round(float(monte_carlo_5th_pct), 4),
        }

    def run(self) -> dict:
        total = float(
            sum(float(p.get("market_value", 0.0) or 0.0) for p in (self.current_portfolio or []))
        )
        if total == 0:
            current_weights = {}
        else:
            current_weights = {
                str(p.get("ticker", "")).upper().strip(): float(p.get("market_value", 0.0) or 0.0)
                / total
                for p in (self.current_portfolio or [])
                if str(p.get("ticker", "")).strip()
            }

        before = self._compute_metrics(current_weights, "before")
        after = self._compute_metrics(self.proposed_weights, "after")
        return {"before": before, "after": after}


def validate_cro_claims(before: dict, after: dict) -> list[str]:
    warnings: list[str] = []

    if not before or not after:
        msg = "WARNING: Missing before/after metrics; cannot validate CRO claims."
        logger.warning(msg)
        return [msg]

    # CHECK 1 — VaR improvement
    before_var = before.get("var_95")
    after_var = after.get("var_95")
    if before_var is not None and after_var is not None and before_var != 0:
        var_improvement = float(before_var) - float(after_var)
        improvement_pct = abs(var_improvement) / abs(float(before_var))
        if improvement_pct < 0.15:
            warnings.append(
                "WARNING: VaR improvement less than 15% "
                f"(actual: {improvement_pct:.1%}). "
                "CRO rebalancing may not meaningfully "
                "reduce tail risk."
            )

    # CHECK 2 — Sharpe degradation
    before_sharpe = before.get("sharpe")
    after_sharpe = after.get("sharpe")
    if (
        before_sharpe is not None
        and after_sharpe is not None
        and float(before_sharpe) > 0
    ):
        degradation = (float(before_sharpe) - float(after_sharpe)) / abs(float(before_sharpe))
        if degradation > 0.30:
            warnings.append(
                "WARNING: Sharpe ratio degrades by "
                f"{degradation:.1%} after rebalance. "
                "Defensive positioning significantly "
                "reduces risk-adjusted return."
            )

    # CHECK 3 — Monte Carlo improvement
    before_mc = before.get("monte_carlo_5th_pct")
    after_mc = after.get("monte_carlo_5th_pct")
    if before_mc is not None and after_mc is not None:
        mc_improvement = float(after_mc) - float(before_mc)
        if mc_improvement < 0.05:
            warnings.append(
                "WARNING: Monte Carlo 5th percentile "
                f"improves only {mc_improvement:.1%}. "
                "Proposed rebalance provides limited "
                "downside protection improvement."
            )

    if warnings:
        for w in warnings:
            logger.warning(w)
    else:
        logger.info("Post-rebalance validation passed")
    return warnings
