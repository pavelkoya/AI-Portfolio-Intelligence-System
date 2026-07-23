from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from config.settings import MONTE_CARLO_SIMULATIONS as N_SIMULATIONS
from config.settings import RISK_FREE_RATE, TRADING_DAYS

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class PortfolioOptimizer:
    """
    Portfolio Monte Carlo simulation, correlation inspection, and PyPortfolioOpt allocation.
    """

    def __init__(
        self,
        prices: Dict[str, pd.DataFrame],
        portfolio: List[Dict[str, Any]],
        n_simulations: int = N_SIMULATIONS,
        trading_days: int = TRADING_DAYS,
    ):
        self.prices = prices
        self.portfolio = portfolio or []
        self.n_simulations = int(n_simulations)
        self.trading_days = int(trading_days)

    def _portfolio_tickers(self) -> List[str]:
        tickers: List[str] = []
        seen: set[str] = set()
        for pos in self.portfolio:
            t = str(pos.get("ticker", "")).upper().strip()
            if not t or t in seen:
                continue
            if t in self.prices:
                tickers.append(t)
                seen.add(t)
        return tickers

    def _total_market_value(self) -> float:
        total = 0.0
        for pos in self.portfolio:
            mv = pos.get("market_value")
            if mv is None:
                continue
            try:
                total += float(mv)
            except (TypeError, ValueError):
                continue
        return total

    def _current_weights(self) -> Dict[str, float]:
        tickers = self._portfolio_tickers()
        total = self._total_market_value()
        if total <= 0:
            return {t: 0.0 for t in tickers}

        weights: Dict[str, float] = {}
        for t in tickers:
            mv = 0.0
            for pos in self.portfolio:
                if str(pos.get("ticker", "")).upper().strip() == t:
                    try:
                        mv = float(pos.get("market_value", 0.0))
                    except (TypeError, ValueError):
                        mv = 0.0
                    break
            weights[t] = mv / total
        return weights

    def _aligned_returns_df(self, tickers: List[str]) -> pd.DataFrame:
        """
        Create a returns DataFrame with columns = `tickers`, aligned on common dates (inner join).
        """
        close_df = pd.concat(
            {
                t: self.prices[t].set_index("date")["close"]
                for t in tickers
                if t in self.prices and self.prices[t] is not None and not self.prices[t].empty
            },
            axis=1,
        ).dropna(how="all")
        if close_df.empty:
            return pd.DataFrame()

        returns = close_df.sort_index().pct_change().dropna(how="any")
        return returns

    def monte_carlo(self) -> Dict[str, Any]:
        """
        Simulate portfolio value over `self.trading_days` using correlated return simulation.
        """
        logger.info(
            "PortfolioOptimizer.monte_carlo: n_simulations=%d trading_days=%d",
            self.n_simulations,
            self.trading_days,
        )
        tickers = self._portfolio_tickers()
        if not tickers:
            logger.warning("PortfolioOptimizer.monte_carlo: no tickers from portfolio")
            return {
                "simulations": np.zeros((self.n_simulations, self.trading_days), dtype=float),
                "initial_value": 0.0,
                "percentile_5": float("nan"),
                "percentile_50": float("nan"),
                "percentile_95": float("nan"),
                "expected_return_pct": float("nan"),
                "var_95_dollar": float("nan"),
            }

        weights = self._current_weights()
        weight_vec = np.array([weights[t] for t in tickers], dtype=float)
        total_value = self._total_market_value()

        rets = self._aligned_returns_df(tickers)
        if rets.empty or rets.shape[0] < 2:
            logger.warning(
                "PortfolioOptimizer.monte_carlo: insufficient aligned returns (rows=%d)",
                0 if rets.empty else rets.shape[0],
            )
            sims = np.zeros((self.n_simulations, self.trading_days), dtype=float)
            return {
                "simulations": sims,
                "initial_value": float(total_value),
                "percentile_5": float("nan"),
                "percentile_50": float("nan"),
                "percentile_95": float("nan"),
                "expected_return_pct": float("nan"),
                "var_95_dollar": float("nan"),
            }

        mean_vec = rets.mean(axis=0).values  # daily
        cov_matrix = rets.cov().values

        # Correlated sampling: correlated_returns = mean + L @ Z
        Z = np.random.standard_normal(size=(self.n_simulations, self.trading_days, len(tickers)))

        L = None
        jitter = 0.0
        for attempt in range(6):
            try:
                if jitter > 0:
                    cov_try = cov_matrix + jitter * np.eye(cov_matrix.shape[0])
                else:
                    cov_try = cov_matrix
                L = np.linalg.cholesky(cov_try)
                break
            except np.linalg.LinAlgError:
                jitter = 1e-10 * (10 ** attempt)
        if L is None:
            logger.warning(
                "PortfolioOptimizer.monte_carlo: Cholesky failed; falling back to diagonal covariance"
            )
            cov_matrix = np.diag(np.diag(cov_matrix))
            L = np.linalg.cholesky(cov_matrix + 1e-12 * np.eye(cov_matrix.shape[0]))

        correlated_returns = mean_vec[None, None, :] + np.einsum(
            "ij,sdj->sdi", L, Z
        )  # (n_sims, days, n_tickers)

        port_daily_returns = correlated_returns @ weight_vec  # (n_sims, days)
        simulations = float(total_value) * np.cumprod(1.0 + port_daily_returns, axis=1)

        final_values = simulations[:, -1]
        percentile_5 = float(np.percentile(final_values, 5))
        percentile_50 = float(np.percentile(final_values, 50))
        percentile_95 = float(np.percentile(final_values, 95))
        expected_return_pct = (
            float((percentile_50 - float(total_value)) / float(total_value))
            if total_value > 0
            else float("nan")
        )
        var_95_dollar = float(total_value - percentile_5)

        return {
            "simulations": simulations,
            "initial_value": float(total_value),
            "percentile_5": percentile_5,
            "percentile_50": percentile_50,
            "percentile_95": percentile_95,
            "expected_return_pct": expected_return_pct,
            "var_95_dollar": var_95_dollar,
        }

    def correlation_matrix(self) -> Dict[str, Any]:
        """
        Correlation matrix between tickers using inner-aligned daily returns.
        """
        logger.info("PortfolioOptimizer.correlation_matrix: computing correlations")
        tickers = [str(t).upper().strip() for t in self.prices.keys() if t]
        tickers = [t for t in tickers if t in self.prices]
        tickers = list(dict.fromkeys(tickers))
        if not tickers:
            return {"matrix": pd.DataFrame(), "tickers": [], "highly_correlated": []}

        rets = self._aligned_returns_df(tickers)
        if rets.empty:
            return {"matrix": pd.DataFrame(), "tickers": tickers, "highly_correlated": []}

        corr = rets.corr()
        # Ensure exact 1.0 on the diagonal (makes tests stable)
        diag_idx = np.diag_indices_from(corr.values)
        corr.values[diag_idx] = 1.0

        highly_correlated: List[Tuple[str, str]] = []
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                t1, t2 = tickers[i], tickers[j]
                val = float(corr.loc[t1, t2])
                if val > 0.85:
                    highly_correlated.append((t1, t2))

        # Flag QQQ/QQQM overlap automatically
        if "QQQ" in tickers and "QQQM" in tickers:
            pair = ("QQQ", "QQQM")
            if pair not in highly_correlated and ("QQQM", "QQQ") not in highly_correlated:
                highly_correlated.append(pair)

        return {"matrix": corr, "tickers": tickers, "highly_correlated": highly_correlated}

    def efficient_frontier(self) -> Dict[str, Any]:
        """
        Optimize portfolio weights with PyPortfolioOpt (max Sharpe and min volatility).
        """
        logger.info("PortfolioOptimizer.efficient_frontier: running optimizations")

        current_weights = self._current_weights()
        tickers = list(current_weights.keys())
        if not tickers:
            logger.warning("PortfolioOptimizer.efficient_frontier: no current weights")
            return {
                "max_sharpe_weights": {},
                "min_vol_weights": {},
                "current_weights": {},
                "max_sharpe_performance": {
                    "expected_return": float("nan"),
                    "volatility": float("nan"),
                    "sharpe": float("nan"),
                },
                "rebalancing_needed": {},
            }

        corr_info = self.correlation_matrix()
        matrix = corr_info.get("matrix")
        if isinstance(matrix, pd.DataFrame) and not matrix.empty:
            # Warn if two tickers are nearly identical (QQQ/QQQM-like overlap).
            for i in range(len(tickers)):
                for j in range(i + 1, len(tickers)):
                    t1, t2 = tickers[i], tickers[j]
                    try:
                        if float(matrix.loc[t1, t2]) > 0.97:
                            logger.warning(
                                "PortfolioOptimizer.efficient_frontier: high overlap correlation %.4f between %s and %s",
                                float(matrix.loc[t1, t2]),
                                t1,
                                t2,
                            )
                    except KeyError:
                        continue

        # Build prices_df for PyPortfolioOpt: tickers as columns, date as index
        try:
            prices_df = pd.DataFrame(
                {
                    t: self.prices[t].set_index("date")["close"]
                    for t in tickers
                    if t in self.prices
                }
            ).dropna(how="all")

            prices_df = prices_df.dropna(how="any")
            if prices_df.empty:
                raise ValueError("prices_df empty after alignment")

            from pypfopt import expected_returns, risk_models
            from pypfopt.efficient_frontier import EfficientFrontier

            mu = expected_returns.mean_historical_return(prices_df)
            S = risk_models.sample_cov(prices_df)

            # Instance 1
            ef_sharpe = EfficientFrontier(mu, S)
            ef_sharpe.max_sharpe(risk_free_rate=RISK_FREE_RATE)
            max_weights = ef_sharpe.clean_weights()
            max_expected_return, max_volatility, max_sharpe = ef_sharpe.portfolio_performance(
                risk_free_rate=RISK_FREE_RATE, verbose=False
            )

            # Instance 2 — fresh, never touched
            ef_minvol = EfficientFrontier(mu, S)
            ef_minvol.min_volatility()
            min_weights = ef_minvol.clean_weights()

            rebalancing_needed: Dict[str, float] = {}
            all_keys = set(current_weights.keys()) | set(max_weights.keys())
            for t in all_keys:
                rebalancing_needed[t] = float(current_weights.get(t, 0.0) - max_weights.get(t, 0.0))

            return {
                "max_sharpe_weights": {k: float(v) for k, v in max_weights.items()},
                "min_vol_weights": {k: float(v) for k, v in min_weights.items()},
                "current_weights": {k: float(v) for k, v in current_weights.items()},
                "max_sharpe_performance": {
                    "expected_return": float(max_expected_return),
                    "volatility": float(max_volatility),
                    "sharpe": float(max_sharpe),
                },
                "rebalancing_needed": rebalancing_needed,
            }
        except Exception as e:
            logger.exception("PortfolioOptimizer.efficient_frontier: optimization failed: %s", str(e))
            # On failure, return current weights as both optimal sets.
            max_weights = current_weights.copy()
            min_weights = current_weights.copy()
            rebalancing_needed = {t: 0.0 for t in current_weights.keys()}
            return {
                "max_sharpe_weights": {k: float(v) for k, v in max_weights.items()},
                "min_vol_weights": {k: float(v) for k, v in min_weights.items()},
                "current_weights": {k: float(v) for k, v in current_weights.items()},
                "max_sharpe_performance": {
                    "expected_return": float("nan"),
                    "volatility": float("nan"),
                    "sharpe": float("nan"),
                },
                "rebalancing_needed": {k: 0.0 for k in current_weights.keys()},
            }

    def run_all(self) -> Dict[str, Any]:
        logger.info("PortfolioOptimizer.run_all: starting")
        return {
            "monte_carlo": self.monte_carlo(),
            "correlation": self.correlation_matrix(),
            "efficient_frontier": self.efficient_frontier(),
        }
