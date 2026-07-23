from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from pypfopt import HRPOpt

from config.settings import TRADING_DAYS
from data.database import DatabaseManager


class HRPEngine:
    def __init__(
        self,
        prices: dict[str, pd.DataFrame],
        portfolio: list[dict],
        monte_carlo_output: dict = None,
    ):
        self.prices = prices
        self.portfolio = portfolio
        self.mc_output = monte_carlo_output
        self.logger = logging.getLogger(__name__)

    def _build_returns_df(self) -> pd.DataFrame:
        close_series: dict[str, pd.Series] = {}
        for ticker, df in (self.prices or {}).items():
            if df is None or getattr(df, "empty", True):
                continue
            if "date" not in df.columns or "close" not in df.columns:
                continue
            tmp = df.copy()
            tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.normalize()
            tmp["close"] = pd.to_numeric(tmp["close"], errors="coerce")
            tmp = tmp.dropna(subset=["date", "close"]).sort_values("date")
            s = tmp.set_index("date")["close"].astype(float)
            s.name = str(ticker).upper().strip()
            close_series[s.name] = s

        if not close_series:
            return pd.DataFrame()

        close_df = pd.concat(close_series, axis=1).sort_index()
        close_df = close_df.tail(TRADING_DAYS)
        returns = close_df.pct_change().dropna(how="any")
        return returns

    def _get_current_weights(self) -> dict:
        total = 0.0
        for p in self.portfolio or []:
            mv = p.get("market_value")
            if mv is None:
                continue
            try:
                total += float(mv)
            except (TypeError, ValueError):
                continue

        if total <= 0:
            return {}

        out: dict[str, float] = {}
        for p in self.portfolio or []:
            t = str(p.get("ticker", "")).upper().strip()
            if not t:
                continue
            mv = p.get("market_value")
            if mv is None:
                continue
            try:
                out[t] = float(mv) / total
            except (TypeError, ValueError):
                continue
        return out

    def compute_hrp_weights(self) -> dict:
        try:
            returns = self._build_returns_df()
            if returns is None or returns.empty:
                self.logger.warning("HRPEngine: empty returns; falling back to current weights")
                return self._get_current_weights()

            hrp = HRPOpt(returns)
            hrp.optimize()
            weights = hrp.clean_weights()
            return {str(k).upper().strip(): float(v) for k, v in (weights or {}).items()}
        except Exception:
            self.logger.exception("HRPEngine: HRP optimization failed; using current weights")
            return self._get_current_weights()

    def compute_rebalancing_table(self) -> list[dict]:
        hrp_weights = self.compute_hrp_weights()
        current_weights = self._get_current_weights()

        tickers = sorted(set(current_weights.keys()) | set(hrp_weights.keys()))
        rows: list[dict] = []
        for t in tickers:
            cw = float(current_weights.get(t, 0.0) or 0.0)
            hw = float(hrp_weights.get(t, 0.0) or 0.0)
            delta = hw - cw
            if delta > 0.02:
                action = "ADD"
            elif delta < -0.02:
                action = "REDUCE"
            else:
                action = "HOLD"
            rows.append(
                {
                    "ticker": t,
                    "current_weight": round(float(cw), 4),
                    "hrp_weight": round(float(hw), 4),
                    "delta": round(float(delta), 4),
                    "action": action,
                }
            )

        rows.sort(key=lambda r: abs(float(r.get("delta", 0.0) or 0.0)), reverse=True)
        return rows

    def compute_marginal_cvar(self) -> dict:
        if self.mc_output is None:
            self.logger.warning("HRPEngine: monte_carlo_output missing; skipping CVaR")
            return {}

        try:
            sims = self.mc_output.get("simulations")
            initial = self.mc_output.get("initial_value")
            if sims is None or initial is None:
                self.logger.warning("HRPEngine: missing simulations/initial_value; skipping CVaR")
                return {}

            simulations = np.asarray(sims, dtype=float)
            if simulations.ndim != 2 or simulations.shape[1] < 2:
                self.logger.warning("HRPEngine: unexpected simulations shape; skipping CVaR")
                return {}

            initial_value = float(initial)
            final = simulations[:, -1]
            returns = (final - initial_value) / initial_value
            var_threshold = np.percentile(returns, 5)
            cvar_scenarios = returns[returns <= var_threshold]
            if cvar_scenarios.size == 0:
                portfolio_cvar = float("nan")
            else:
                portfolio_cvar = float(np.mean(cvar_scenarios))

            current_weights = self._get_current_weights()
            marginal = {t: float(w) * float(portfolio_cvar) for t, w in current_weights.items()}

            concentration_flags: dict[str, bool] = {}
            worst = None
            if marginal:
                worst = min(marginal, key=marginal.get)
                concentration_flags = {t: (t == worst) for t in marginal}

            return {
                "marginal_cvar": marginal,
                "concentration_flags": concentration_flags,
                "portfolio_cvar": round(float(portfolio_cvar), 4)
                if np.isfinite(portfolio_cvar)
                else float("nan"),
            }
        except Exception:
            self.logger.exception("HRPEngine: CVaR computation failed")
            return {}

    def get_cluster_map(self) -> dict:
        returns_df = self._build_returns_df()
        if returns_df is None or returns_df.empty or returns_df.shape[1] == 0:
            return {}

        corr = returns_df.corr()
        out: dict[str, str] = {}
        for t in corr.columns:
            row = corr.loc[t].drop(labels=[t], errors="ignore")
            if row.empty:
                out[t] = t
                continue
            peer = row.dropna().idxmax() if not row.dropna().empty else None
            out[t] = str(peer) if peer is not None else t
        return out

    def run(self) -> dict:
        hrp_weights = self.compute_hrp_weights()
        rebalancing = self.compute_rebalancing_table()
        cvar = self.compute_marginal_cvar()
        cluster_map = self.get_cluster_map()
        current_weights = self._get_current_weights()

        actions = {"ADD": 0, "REDUCE": 0, "HOLD": 0}
        for r in rebalancing:
            a = r.get("action")
            if a in actions:
                actions[a] += 1

        self.logger.info(
            "HRP complete. Actions: ADD=%d, REDUCE=%d, HOLD=%d",
            actions["ADD"],
            actions["REDUCE"],
            actions["HOLD"],
        )

        return {
            "hrp_weights": hrp_weights,
            "rebalancing_table": rebalancing,
            "cvar_analysis": cvar,
            "cluster_map": cluster_map,
            "current_weights": current_weights,
        }

    def get_committee_input(self) -> dict:
        hrp_weights = self.compute_hrp_weights()
        rebalancing_table = self.compute_rebalancing_table()
        cvar = self.compute_marginal_cvar()

        worst_ticker = None
        flags = (cvar or {}).get("concentration_flags") or {}
        for t, is_worst in flags.items():
            if is_worst:
                worst_ticker = t
                break

        return {
            "rebalancing_table": rebalancing_table,
            "concentration_risk_ticker": worst_ticker,
            "portfolio_cvar": (cvar or {}).get("portfolio_cvar"),
            "hrp_weights": hrp_weights,
        }

