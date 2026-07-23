import logging

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from config.settings import TRADING_DAYS


REGIME_FACTOR_WEIGHTS = {
    "Bull": {
        "wM": 0.40,  # momentum matters most
        "wV": 0.20,  # low vol matters less
        "wC": 0.10,  # correlation less critical
        "wT": 0.30,  # trend breakout important
    },
    "Neutral": {
        "wM": 0.25,  # equal weights (RAAM default)
        "wV": 0.25,
        "wC": 0.25,
        "wT": 0.25,
    },
    "Bear": {
        "wM": 0.10,  # momentum unreliable in Bear
        "wV": 0.50,  # avoid high-vol assets
        "wC": 0.30,  # avoid correlated assets
        "wT": 0.10,  # trend less reliable
    },
}


class RAAMEngine:
    def __init__(
        self,
        prices: dict,
        garch_output: dict,
        correlation_matrix: dict,
        risk_levels: dict,
        trend_output: dict,
        regime_current: dict,
        top_n: int = 5,
        momentum_lookback: int = 84,
        use_regime_weights: bool = True,
    ):
        self.prices = prices
        self.garch_output = garch_output
        self.correlation_matrix = correlation_matrix
        self.risk_levels = risk_levels
        self.trend_output = trend_output
        self.regime_current = regime_current

        self.top_n = top_n
        self.momentum_lookback = momentum_lookback
        self.use_regime_weights = use_regime_weights

        self.logger = logging.getLogger(__name__)
        self.results = {}
        self.tickers = list(prices.keys())
        # Imported by request context; kept to make annualization conventions explicit.
        self.trading_days = TRADING_DAYS

    def _compute_momentum(self) -> dict:
        """
        Compute 4-month absolute momentum per ticker.
        ROC = (close[-1] / close[-lookback] - 1)
        Return {ticker: float}
        """
        momentum = {}
        for ticker, df in self.prices.items():
            try:
                closes = df["close"].dropna().values
                if len(closes) < self.momentum_lookback:
                    momentum[ticker] = 0.0
                    self.logger.warning(
                        "RAAM: %s has only %d rows for momentum (need %d)",
                        ticker,
                        len(closes),
                        self.momentum_lookback,
                    )
                    continue
                roc = closes[-1] / closes[-self.momentum_lookback] - 1
                momentum[ticker] = float(roc)
            except Exception as e:
                self.logger.warning("RAAM: momentum failed %s: %s", ticker, e)
                momentum[ticker] = 0.0
        return momentum

    def _compute_vol_scores(self) -> dict:
        """
        Extract annualized vol from GARCH output.
        Lower vol = better rank (defensive asset).
        Return {ticker: float}
        """
        scores = {}
        for ticker in self.tickers:
            g = self.garch_output.get(ticker) or {}
            vol = g.get("annualized_vol", None)
            if vol is None:
                vol = 0.20  # fallback: 20% annualized
                self.logger.warning(
                    "RAAM: no GARCH vol for %s, using 20%% fallback",
                    ticker,
                )
            scores[ticker] = float(vol)
        return scores

    def _compute_correlation_scores(self) -> dict:
        """
        Compute average pairwise correlation per ticker.
        Lower avg correlation = better rank (diversifier).
        correlation_matrix is dict-of-dicts:
          {ticker1: {ticker2: float, ...}, ...}
        Return {ticker: float}
        """
        scores = {}
        matrix = self.correlation_matrix
        for ticker in self.tickers:
            row = matrix.get(ticker, {})
            if not row:
                scores[ticker] = 0.5
                continue

            corr_vals = [
                v for t, v in row.items()
                if t != ticker and v is not None
            ]
            scores[ticker] = float(np.mean(corr_vals)) if corr_vals else 0.5
        return scores

    def _compute_atr_trend_signal(self) -> dict:
        """
        Compute ATR trend signal T per ticker.
        RAAM definition:
          +2 if price > rolling_max + ATR (uptrend)
          -2 if price < rolling_min - ATR (downtrend)
           0 otherwise (neutral)
        """
        signals = {}
        for ticker, df in self.prices.items():
            try:
                risk = self.risk_levels.get(ticker) or {}
                atr = risk.get("atr_14", 0) or 0

                closes = df["close"].dropna()
                if len(closes) < 42:
                    signals[ticker] = 0
                    continue

                recent = closes.tail(42)
                high_42 = recent.max()
                low_42 = recent.min()
                price = float(closes.iloc[-1])

                upper = high_42 + atr
                lower = low_42 - atr

                if price > upper:
                    signals[ticker] = 2
                elif price < lower:
                    signals[ticker] = -2
                else:
                    signals[ticker] = 0
            except Exception as e:
                self.logger.warning("RAAM: ATR signal failed %s: %s", ticker, e)
                signals[ticker] = 0
        return signals

    def _get_factor_weights(self) -> dict:
        """
        Get factor weights based on current regime.
        If use_regime_weights=False -> Neutral weights.
        """
        if not self.use_regime_weights:
            return REGIME_FACTOR_WEIGHTS["Neutral"]

        regime = self.regime_current.get("dominant_regime", "Neutral")
        weights = REGIME_FACTOR_WEIGHTS.get(
            regime,
            REGIME_FACTOR_WEIGHTS["Neutral"],
        )
        self.logger.info(
            "RAAM: regime=%s -> wM=%.2f wV=%.2f wC=%.2f wT=%.2f",
            regime,
            weights["wM"],
            weights["wV"],
            weights["wC"],
            weights["wT"],
        )
        return weights

    def _rank_ascending(self, scores: dict, higher_is_better: bool = True) -> dict:
        """
        Rank tickers. Returns {ticker: rank} where rank 1 = best, N = worst.
        Ties handled with average method.
        """
        tickers = list(scores.keys())
        values = [scores[t] for t in tickers]
        if higher_is_better:
            raw_ranks = rankdata([-v for v in values], method="average")
        else:
            raw_ranks = rankdata(values, method="average")
        return {t: float(r) for t, r in zip(tickers, raw_ranks)}

    def run(self) -> dict:
        """
        Main RAAM computation.
        Returns full results dict.
        """
        self.logger.info("RAAM: starting for %d tickers", len(self.tickers))

        # Step 1: compute raw factor scores
        momentum = self._compute_momentum()
        vol = self._compute_vol_scores()
        corr = self._compute_correlation_scores()
        trend_t = self._compute_atr_trend_signal()

        # Step 2: rank each factor
        rank_M = self._rank_ascending(momentum, higher_is_better=True)
        rank_V = self._rank_ascending(vol, higher_is_better=False)
        rank_C = self._rank_ascending(corr, higher_is_better=False)

        # Step 3: get regime-conditional weights
        weights = self._get_factor_weights()
        wM = weights["wM"]
        wV = weights["wV"]
        wC = weights["wC"]
        wT = weights["wT"]

        # Step 4: compute composite RAAM score
        n = len(self.tickers)
        x = 100

        total_rank = {}
        for ticker in self.tickers:
            score = (
                wM * rank_M.get(ticker, n / 2)
                + wV * rank_V.get(ticker, n / 2)
                + wC * rank_C.get(ticker, n / 2)
                - wT * trend_t.get(ticker, 0)
                + momentum.get(ticker, 0) / x
            )
            total_rank[ticker] = score

        # Step 5: sort ascending -> lower = better
        sorted_tickers = sorted(total_rank.items(), key=lambda x_: x_[1])

        # Step 6: select top N with momentum filter
        selected = []
        raam_weights = {}

        for ticker, _score in sorted_tickers[: self.top_n]:
            mom = momentum.get(ticker, 0)
            if mom > 0:
                selected.append(ticker)
            else:
                self.logger.info(
                    "RAAM: %s has negative momentum %.2f%% -> replaced with CASH",
                    ticker,
                    mom * 100,
                )

        n_selected = len(selected)
        if n_selected > 0:
            equal_w = 1.0 / self.top_n
            for t in selected:
                raam_weights[t] = equal_w
            cash_pct = (self.top_n - n_selected) / self.top_n
            if cash_pct > 0:
                raam_weights["CASH"] = cash_pct
        else:
            raam_weights["CASH"] = 1.0
            self.logger.warning(
                "RAAM: all tickers have negative momentum - 100%% cash"
            )

        # Step 7: log results
        self.logger.info(
            "RAAM: Top %d selected: %s",
            self.top_n,
            [f"{t}({total_rank[t]:.3f})" for t in selected],
        )
        self.logger.info(
            "RAAM: Weights: %s",
            {t: f"{w:.1%}" for t, w in raam_weights.items()},
        )

        # Step 8: build full output
        factor_breakdown = {}
        for ticker in self.tickers:
            factor_breakdown[ticker] = {
                "momentum_pct": round(momentum.get(ticker, 0) * 100, 2),
                "garch_vol": round(vol.get(ticker, 0), 4),
                "avg_corr": round(corr.get(ticker, 0), 4),
                "atr_signal": trend_t.get(ticker, 0),
                "rank_M": int(rank_M.get(ticker, 0)),
                "rank_V": int(rank_V.get(ticker, 0)),
                "rank_C": int(rank_C.get(ticker, 0)),
                "total_rank": round(total_rank.get(ticker, 0), 4),
                "selected": ticker in selected,
            }

        self.results = {
            "selected_tickers": selected,
            "raam_weights": raam_weights,
            "all_rankings": dict(sorted_tickers),
            "factor_scores": {
                "momentum": momentum,
                "volatility": vol,
                "correlation": corr,
                "atr_signal": trend_t,
            },
            "factor_ranks": {
                "rank_M": rank_M,
                "rank_V": rank_V,
                "rank_C": rank_C,
            },
            "factor_breakdown": factor_breakdown,
            "regime_weights_used": weights,
            "regime": self.regime_current.get("dominant_regime", "Neutral"),
            "top_n": self.top_n,
            "n_selected": n_selected,
            "cash_pct": raam_weights.get("CASH", 0),
        }
        return self.results

    def get_committee_input(self) -> dict:
        """Lean dict for LLM consumption."""
        if not self.results:
            self.run()
        bd = self.results.get("factor_breakdown", {})
        return {
            "raam_selected": self.results.get("selected_tickers", []),
            "raam_weights": self.results.get("raam_weights", {}),
            "raam_regime_weights": self.results.get("regime_weights_used", {}),
            "raam_top_rankings": {
                t: {
                    "total_rank": bd[t]["total_rank"],
                    "momentum": bd[t]["momentum_pct"],
                    "selected": bd[t]["selected"],
                }
                for t in list(bd.keys())[:8]
            },
        }
