from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from quant.technical import TechnicalAnalyzer
from quant.risk import RiskAnalyzer
from quant.portfolio import PortfolioOptimizer

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class QuantEngine:
    """
    Orchestrates Phase 3 quant computations from Phase 1 cleaned data.
    """

    def __init__(self, data: Dict[str, Any]):
        _prices = data.get("prices")
        self.prices: Dict[str, pd.DataFrame] = _prices if isinstance(_prices, dict) else {}

        _portfolio = data.get("portfolio")
        self.portfolio: List[Dict[str, Any]] = _portfolio if isinstance(_portfolio, list) else []

        _benchmark = data.get("benchmark")
        self.benchmark: pd.DataFrame = (
            _benchmark
            if (
                _benchmark is not None
                and not (isinstance(_benchmark, pd.DataFrame) and _benchmark.empty)
            )
            else pd.DataFrame()
        )
        if not isinstance(self.benchmark, pd.DataFrame):
            self.benchmark = pd.DataFrame()

        # Analyzers
        self.technical_analyzer = TechnicalAnalyzer(self.prices)
        self.risk_analyzer = RiskAnalyzer(
            prices=self.prices,
            benchmark=self.benchmark,
            portfolio=self.portfolio,
        )
        self.portfolio_optimizer = PortfolioOptimizer(
            prices=self.prices,
            portfolio=self.portfolio,
        )

    def run(self) -> Dict[str, Any]:
        start = time.time()
        logger.info("QuantEngine.run: starting")
        try:
            technical = self.technical_analyzer.run_all()
            risk = self.risk_analyzer.run_all()
            portfolio_metrics = self.portfolio_optimizer.run_all()
            return {
                "technical": technical,
                "risk": risk,
                "portfolio": portfolio_metrics,
            }
        finally:
            elapsed = time.time() - start
            logger.info("QuantEngine.run: finished in %.3fs", elapsed)

    @staticmethod
    def _strip_series_or_array(value: Any) -> Any:
        """
        Convert large objects (Series/ndarray) into scalars where possible.
        """
        if value is None:
            return None
        # Series -> last scalar if possible
        if isinstance(value, pd.Series):
            if value.dropna().empty:
                return None
            return float(value.dropna().iloc[-1])
        # ndarray -> drop (keep None)
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return None
        except Exception:
            pass
        return value

    def get_committee_input(self) -> Dict[str, Any]:
        """
        Format a lean metrics dict for the Phase 4 Multi-Agent Committee.
        """
        metrics = self.run()

        technical: Dict[str, Any] = metrics.get("technical") or {}
        risk: Dict[str, Any] = metrics.get("risk") or {}
        portfolio: Dict[str, Any] = metrics.get("portfolio") or {}

        portfolio_sharpe_obj = risk.get("portfolio", {}).get("sharpe") or {}
        portfolio_beta_obj = risk.get("portfolio", {}).get("beta_alpha") or {}
        portfolio_mdd_obj = risk.get("portfolio", {}).get("max_drawdown") or {}
        portfolio_var_obj = risk.get("portfolio", {}).get("var_95") or {}

        mc = portfolio.get("monte_carlo") or {}
        ef = portfolio.get("efficient_frontier") or {}
        corr = portfolio.get("correlation") or {}

        # Per-ticker: merge technical and risk for only required scalar keys
        per_ticker: Dict[str, Any] = {}
        all_tickers = sorted([t for t in self.prices.keys() if t])
        for ticker in all_tickers:
            t_tech = technical.get(ticker) or {}
            t_rsi = t_tech.get("rsi") or {}
            t_macd = t_tech.get("macd") or {}
            t_ma = t_tech.get("ma") or {}

            t_risk = risk.get(ticker) or {}
            t_sharpe_obj = t_risk.get("sharpe") or {}
            t_mdd_obj = t_risk.get("max_drawdown") or {}
            t_beta_obj = t_risk.get("beta_alpha") or {}

            per_ticker[ticker] = {
                "rsi": float(t_rsi.get("rsi_current")) if t_rsi.get("rsi_current") is not None else None,
                "rsi_signal": t_rsi.get("signal"),
                "macd_signal": t_macd.get("signal"),
                "above_sma_200": bool(t_ma.get("above_sma_200")) if t_ma.get("above_sma_200") is not None else None,
                "sharpe": float(t_sharpe_obj.get("sharpe")) if t_sharpe_obj.get("sharpe") is not None else None,
                "max_drawdown_pct": t_mdd_obj.get("max_drawdown_pct"),
                "beta": float(t_beta_obj.get("beta")) if t_beta_obj.get("beta") is not None else None,
            }

        return {
            "portfolio_sharpe": float(portfolio_sharpe_obj.get("sharpe"))
            if portfolio_sharpe_obj.get("sharpe") is not None
            else None,
            "portfolio_max_drawdown_pct": portfolio_mdd_obj.get("max_drawdown_pct"),
            "portfolio_var_95_dollar": float(portfolio_var_obj.get("var_95_dollar"))
            if portfolio_var_obj.get("var_95_dollar") is not None
            else None,
            "portfolio_beta": float(portfolio_beta_obj.get("beta"))
            if portfolio_beta_obj.get("beta") is not None
            else None,
            "portfolio_alpha": float(portfolio_beta_obj.get("alpha"))
            if portfolio_beta_obj.get("alpha") is not None
            else None,
            "monte_carlo": {
                "initial_value": float(mc.get("initial_value"))
                if mc.get("initial_value") is not None
                else None,
                "percentile_5": float(mc.get("percentile_5"))
                if mc.get("percentile_5") is not None
                else None,
                "percentile_50": float(mc.get("percentile_50"))
                if mc.get("percentile_50") is not None
                else None,
                "percentile_95": float(mc.get("percentile_95"))
                if mc.get("percentile_95") is not None
                else None,
                "expected_return_pct": float(mc.get("expected_return_pct"))
                if mc.get("expected_return_pct") is not None
                else None,
            },
            "optimal_weights": ef.get("max_sharpe_weights") or {},
            "current_weights": ef.get("current_weights") or {},
            "rebalancing_needed": ef.get("rebalancing_needed") or {},
            "highly_correlated_pairs": corr.get("highly_correlated") or [],
            "per_ticker": per_ticker,
        }
