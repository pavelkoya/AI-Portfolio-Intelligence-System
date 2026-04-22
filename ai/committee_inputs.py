from __future__ import annotations

import json
import logging
from datetime import datetime


class CommitteeInputs:
    def __init__(
        self,
        quant_committee_input: dict,
        hrp_committee_input: dict,
        regime_output: dict,
        garch_committee_input: dict,
        stop_levels: dict,
        analyst_targets: dict,
        portfolio: list[dict],
    ):
        self.quant_ci = quant_committee_input or {}
        self.hrp_ci = hrp_committee_input or {}
        self.regime_output = regime_output or {}
        self.garch_ci = garch_committee_input or {}
        self.stop_levels = stop_levels or {}
        self.analyst_targets = analyst_targets or {}
        self.portfolio = portfolio or []

        self.logger = logging.getLogger(__name__)

    def build(self) -> dict:
        total_value = sum(p["market_value"] for p in self.portfolio)
        return {
            "as_of": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "portfolio_summary": {
                "total_value": round(total_value, 2),
                "position_count": len(self.portfolio),
                "positions": [
                    {
                        "ticker": p["ticker"],
                        "market_value": p["market_value"],
                        "current_price": p["current_price"],
                        "avg_buy_price": p["average_buy_price"],
                        "unrealized_pnl": p.get("unrealized_pnl"),
                    }
                    for p in self.portfolio
                ],
            },
            "regime": self.regime_output,
            "risk_metrics": {
                "portfolio_sharpe": self.quant_ci.get("portfolio_sharpe"),
                "portfolio_max_drawdown": self.quant_ci.get("portfolio_max_drawdown_pct"),
                "portfolio_var_95_dollar": self.quant_ci.get("portfolio_var_95_dollar"),
                "portfolio_beta": self.quant_ci.get("portfolio_beta"),
                "portfolio_alpha": self.quant_ci.get("portfolio_alpha"),
                "monte_carlo": self.quant_ci.get("monte_carlo"),
            },
            "optimization": {
                "optimal_weights": self.quant_ci.get("optimal_weights"),
                "hrp_rebalancing": self.hrp_ci.get("rebalancing_table"),
                "concentration_risk_ticker": self.hrp_ci.get(
                    "concentration_risk_ticker"
                ),
                "portfolio_cvar": self.hrp_ci.get("portfolio_cvar"),
            },
            "per_ticker": {
                ticker: {
                    "rsi": data.get("rsi"),
                    "rsi_signal": data.get("rsi_signal"),
                    "macd_signal": data.get("macd_signal"),
                    "above_sma_200": data.get("above_sma_200"),
                    "sharpe": data.get("sharpe"),
                    "max_drawdown_pct": data.get("max_drawdown_pct"),
                    "beta": data.get("beta"),
                    "garch_vol": (self.garch_ci.get(ticker, {}) or {}).get(
                        "annualized_vol"
                    ),
                    "vol_regime": (self.garch_ci.get(ticker, {}) or {}).get("vol_regime"),
                    "stop_loss": (self.stop_levels.get(ticker, {}) or {}).get("stop_loss"),
                    "take_profit": (self.stop_levels.get(ticker, {}) or {}).get(
                        "take_profit"
                    ),
                    "analyst_target": (self.analyst_targets.get(ticker, {}) or {}).get(
                        "target_price"
                    ),
                    "analyst_upside": (self.analyst_targets.get(ticker, {}) or {}).get(
                        "upside_pct_str"
                    ),
                    "analyst_signal": (self.analyst_targets.get(ticker, {}) or {}).get(
                        "signal"
                    ),
                }
                for ticker, data in (self.quant_ci.get("per_ticker", {}) or {}).items()
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.build(), indent=2, default=str)

    def build_lean(self) -> dict:
        full = self.build()

        # Keep per_ticker but drop verbose fields
        lean_per_ticker = {}
        for ticker, data in full["per_ticker"].items():
            lean_per_ticker[ticker] = {k: v for k, v in data.items() if v is not None}

        return {
            "as_of": full["as_of"],
            "regime": full["regime"],
            "risk_metrics": full["risk_metrics"],
            "optimization": {
                "hrp_rebalancing": full["optimization"]["hrp_rebalancing"],
                "concentration_risk_ticker": full["optimization"][
                    "concentration_risk_ticker"
                ],
            },
            "per_ticker": lean_per_ticker,
            "portfolio_summary": {
                "total_value": full["portfolio_summary"]["total_value"],
                "position_count": full["portfolio_summary"]["position_count"],
            },
        }
