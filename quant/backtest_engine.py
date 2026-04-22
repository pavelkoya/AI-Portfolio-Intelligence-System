from __future__ import annotations

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd

from config.settings import RISK_FREE_RATE, TRADING_DAYS
from quant.hrp_engine import HRPEngine


def compute_period_metrics(returns: pd.Series, label: str) -> dict:
    """
    Compute Sharpe, max drawdown, total return
    from a daily return Series.
    """
    if returns is None or len(returns) < 5:
        return {
            "label": label,
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    ann_return = returns.mean() * TRADING_DAYS
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = ((ann_return - RISK_FREE_RATE) / ann_vol) if ann_vol > 0 else 0.0

    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    max_dd = float(dd.min())

    total_return = float(cum.iloc[-1] - 1)

    return {
        "label": label,
        "total_return": round(total_return, 6),
        "sharpe": round(float(sharpe), 4),
        "max_drawdown": round(max_dd, 4),
    }


class BacktestEngine:
    def __init__(
        self,
        prices: dict,
        portfolio: list,
        spy: pd.DataFrame,
        train_days: int = 252,
        test_days: int = 63,
        regime_scalar_cap: float = 0.5,
    ):
        self.prices = prices
        self.portfolio = portfolio
        self.spy = spy
        self.train = train_days
        self.test = test_days
        self.regime_scalar_cap = regime_scalar_cap
        self.logger = logging.getLogger(__name__)
        self.results: list[dict] = []
        self._regime_states = None
        self._regime_dates = None
        self._risk_scalars: dict[pd.Timestamp, float] = {}

    def _build_returns_matrix(self, prices_slice: dict[str, pd.DataFrame]) -> pd.DataFrame:
        series_map: dict[str, pd.Series] = {}
        for ticker, df in (prices_slice or {}).items():
            if df is None or getattr(df, "empty", True):
                continue
            if "date" not in df.columns or "close" not in df.columns:
                continue

            tmp = df.copy()
            tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.normalize()
            tmp["close"] = pd.to_numeric(tmp["close"], errors="coerce")
            tmp = tmp.dropna(subset=["date", "close"]).sort_values("date")
            if tmp.empty:
                continue
            s = tmp.set_index("date")["close"].astype(float)
            s.name = str(ticker).upper().strip()
            series_map[s.name] = s

        if not series_map:
            return pd.DataFrame()

        close_df = pd.concat(series_map.values(), axis=1, join="inner")
        close_df.columns = list(series_map.keys())
        returns = close_df.pct_change().dropna(how="any")
        return returns.sort_index()

    def _fit_regime_once(self) -> None:
        from quant.regime_engine import RegimeEngine

        engine = RegimeEngine(self.spy)
        engine.fit()

        if engine.model is None or engine.features is None:
            self.logger.warning("Regime: model/features unavailable, using neutral fallback")
            self._regime_states = np.array([])
            self._regime_dates = pd.DatetimeIndex([])
            self._risk_scalars = {}
            return

        all_states = engine.model.predict(engine.features)
        probs = engine.model.predict_proba(engine.features)

        feature_dates = getattr(engine, "_feature_dates", None)
        if feature_dates is None:
            self.logger.warning("Regime: feature dates unavailable, using neutral fallback")
            self._regime_states = np.array([])
            self._regime_dates = pd.DatetimeIndex([])
            self._risk_scalars = {}
            return

        dates = pd.DatetimeIndex(pd.to_datetime(feature_dates).normalize())
        self._regime_states = all_states
        self._regime_dates = dates

        # Identify crash/bear states from ascending mean returns (same logic as RegimeEngine).
        state_means = getattr(engine, "_state_means", {}) or {}
        if state_means:
            ordered_states = sorted(
                state_means.keys(),
                key=lambda k: np.nan_to_num(state_means[k], nan=-1e9),
            )
        else:
            # Fallback if internals are missing: derive from fitted sequence using aligned returns.
            aligned_returns = getattr(engine, "_spy_returns_aligned", pd.Series(dtype=float))
            ordered_states = sorted(
                set(int(s) for s in all_states.tolist()),
                key=lambda s: float(
                    np.nanmean(aligned_returns.values[all_states == s])
                    if len(aligned_returns) == len(all_states)
                    else -1e9
                ),
            )

        crash_state = int(ordered_states[0]) if ordered_states else 0
        bear_state = int(ordered_states[1]) if len(ordered_states) > 1 else crash_state

        risk_scalars: dict[pd.Timestamp, float] = {}
        for idx, dt in enumerate(dates):
            bear_prob = float(probs[idx, bear_state])
            crash_prob = float(probs[idx, crash_state])
            risk_scalar = (bear_prob * 0.5) + (crash_prob * 1.0)
            risk_scalar = min(max(risk_scalar, 0.0), 1.0)
            risk_scalars[pd.Timestamp(dt)] = float(risk_scalar)

        self._risk_scalars = risk_scalars
        self.logger.info(
            "Regime: fitted once on %d SPY rows, %d dates mapped",
            len(self.spy) if self.spy is not None else 0,
            len(self._risk_scalars),
        )

    def _get_risk_scalar_at(self, window_end_date: pd.Timestamp) -> float:
        if not self._risk_scalars:
            return 0.5

        ts = pd.Timestamp(window_end_date).normalize()
        if ts in self._risk_scalars:
            return float(self._risk_scalars[ts])

        eligible = [d for d in self._risk_scalars.keys() if d <= ts]
        if not eligible:
            return 0.5
        nearest = max(eligible)
        return float(self._risk_scalars[nearest])

    def _apply_regime_to_weights(self, hrp_weights: dict, risk_scalar: float) -> dict:
        # Cap for backtest realism
        capped_scalar = min(risk_scalar, self.regime_scalar_cap)
        equity_fraction = 1.0 - capped_scalar
        adjusted = {t: float(w) * equity_fraction for t, w in (hrp_weights or {}).items()}
        adjusted["CASH"] = capped_scalar
        return adjusted

    def _equal_weight(self, tickers: list[str]) -> dict:
        clean = [str(t).upper().strip() for t in tickers if t]
        if not clean:
            return {}
        w = 1.0 / len(clean)
        return {t: w for t in clean}

    def _compute_weighted_returns(self, returns_df: pd.DataFrame, weights: dict) -> pd.Series:
        if returns_df is None or returns_df.empty or not weights:
            return pd.Series(dtype=float)

        cash_weight = float(weights.get("CASH", 0.0) or 0.0)
        cash_weight = min(max(cash_weight, 0.0), 1.0)

        overlap = [t for t in returns_df.columns if t in weights and t != "CASH"]
        if not overlap:
            return pd.Series(0.0, index=returns_df.index, dtype=float)

        equity_weights = {t: float(weights[t]) for t in overlap}
        eq_sum = float(sum(equity_weights.values()))
        if eq_sum <= 0:
            return pd.Series(0.0, index=returns_df.index, dtype=float)

        # Normalize inside equity sleeve, then apply cash drag on sleeve notional.
        normalized = {t: w / eq_sum for t, w in equity_weights.items()}
        equity_returns = sum(normalized[t] * returns_df[t].astype(float) for t in overlap)
        portfolio_returns = equity_returns * (1.0 - cash_weight)
        return portfolio_returns.astype(float)

    def run(self) -> dict:
        self._fit_regime_once()
        self.results = []

        if self.spy is None or getattr(self.spy, "empty", True):
            self.logger.error("Insufficient history")
            return self._empty_result()

        # Use SPY as the master date index
        # SPY has 5y of history; tickers have 2y
        # We walk forward over SPY dates; for each window
        # we use whatever ticker data is available
        spy_df = self.spy.copy()
        spy_df["date"] = pd.to_datetime(spy_df["date"]).dt.normalize()
        spy_df = spy_df.sort_values("date")
        all_dates = spy_df["date"].tolist()

        self.logger.info(
            "Backtest: SPY date range %s → %s (%d days)",
            str(all_dates[0])[:10] if all_dates else "N/A",
            str(all_dates[-1])[:10] if all_dates else "N/A",
            len(all_dates),
        )
        expected = max(0, (len(all_dates) - self.train) // self.test)
        self.logger.info("Backtest: expecting ~%d periods", expected)
        self.logger.info(
            "Backtest: all_dates range %s → %s (%d total trading days)",
            str(all_dates[0])[:10] if all_dates else "N/A",
            str(all_dates[-1])[:10] if all_dates else "N/A",
            len(all_dates),
        )
        self.logger.info(
            "Backtest: train=%d test=%d max_possible_periods=%d",
            self.train,
            self.test,
            max(0, (len(all_dates) - self.train) // self.test),
        )
        for ticker in list(self.prices.keys())[:3]:
            df = self.prices[ticker]
            self.logger.info(
                "Backtest: %s has %d rows (%s → %s)",
                ticker,
                len(df),
                str(df["date"].min())[:10],
                str(df["date"].max())[:10],
            )

        if len(all_dates) < self.train + self.test:
            self.logger.error("Insufficient history")
            return self._empty_result()

        period_num = 0
        start_idx = 0

        while start_idx + self.train + self.test <= len(all_dates):
            t0_period = time.time()
            train_dates = all_dates[start_idx : start_idx + self.train]
            test_dates = all_dates[start_idx + self.train : start_idx + self.train + self.test]
            window_end = train_dates[-1]
            train_date_set = set(train_dates)
            test_date_set = set(test_dates)

            train_prices: dict[str, pd.DataFrame] = {}
            valid_tickers: list[str] = []
            for ticker, df in (self.prices or {}).items():
                df_copy = df.copy()
                df_copy["date"] = pd.to_datetime(df_copy["date"], errors="coerce").dt.normalize()
                slice_ = df_copy[df_copy["date"].isin(train_date_set)]
                if len(slice_) >= 252:
                    train_prices[ticker] = slice_
                    valid_tickers.append(ticker)
                else:
                    self.logger.debug(
                        "Period %d: %s only has %d train rows - skipping",
                        period_num,
                        ticker,
                        len(slice_),
                    )

            if len(valid_tickers) < 2:
                self.logger.warning(
                    "Period %d: fewer than 2 valid tickers, skipping",
                    period_num,
                )
                start_idx += self.test
                continue

            train_portfolio = [{"ticker": t, "market_value": 1.0} for t in valid_tickers]
            try:
                hrp = HRPEngine(
                    prices=train_prices,
                    portfolio=train_portfolio,
                    monte_carlo_output=None,
                )
                hrp_result = hrp.run()
                hrp_weights = {
                    t: float(w)
                    for t, w in (hrp_result.get("hrp_weights", {}) or {}).items()
                    if t in valid_tickers
                }
            except Exception as e:
                self.logger.warning("Period %d: HRP failed: %s", period_num, e)
                start_idx += self.test
                continue

            # Redistribute among remaining valid names if optimizer dropped any.
            if hrp_weights:
                total_w = float(sum(hrp_weights.values()))
                if total_w > 0:
                    hrp_weights = {t: w / total_w for t, w in hrp_weights.items()}
            else:
                hrp_weights = self._equal_weight(valid_tickers)

            risk_scalar = self._get_risk_scalar_at(pd.Timestamp(window_end))
            adj_weights = self._apply_regime_to_weights(hrp_weights, risk_scalar)
            eq_weights = self._equal_weight(valid_tickers)

            test_prices: dict[str, pd.DataFrame] = {}
            for ticker in valid_tickers:
                df_copy = self.prices[ticker].copy()
                df_copy["date"] = pd.to_datetime(df_copy["date"], errors="coerce").dt.normalize()
                slice_ = df_copy[df_copy["date"].isin(test_date_set)]
                if not slice_.empty:
                    test_prices[ticker] = slice_

            test_returns = self._build_returns_matrix(test_prices)
            if test_returns.empty or len(test_returns) < 5:
                start_idx += self.test
                continue

            hrp_returns = self._compute_weighted_returns(test_returns, adj_weights)
            eq_returns = self._compute_weighted_returns(test_returns, eq_weights)

            hrp_metrics = compute_period_metrics(hrp_returns, "hrp")
            eq_metrics = compute_period_metrics(eq_returns, "benchmark")
            capped = min(risk_scalar, self.regime_scalar_cap)

            self.results.append(
                {
                    "period": period_num,
                    "train_start": str(train_dates[0]),
                    "train_end": str(window_end),
                    "test_start": str(test_dates[0]),
                    "test_end": str(test_dates[-1]),
                    "risk_scalar": round(float(risk_scalar), 3),
                    "risk_scalar_capped": round(float(capped), 3),
                    "n_tickers": len(valid_tickers),
                    "hrp_return": hrp_metrics["total_return"],
                    "hrp_sharpe": hrp_metrics["sharpe"],
                    "hrp_drawdown": hrp_metrics["max_drawdown"],
                    "eq_return": eq_metrics["total_return"],
                    "eq_sharpe": eq_metrics["sharpe"],
                    "eq_drawdown": eq_metrics["max_drawdown"],
                    "regime_bear": bool(risk_scalar > 0.7),
                }
            )

            self.logger.info(
                "  Period %2d | %s → %s | HRP: %+.2f%%  EQ: %+.2f%%  scalar=%.3f(cap=%.1f)  (%.1fs)",
                period_num,
                str(test_dates[0])[:10],
                str(test_dates[-1])[:10],
                hrp_metrics["total_return"] * 100,
                eq_metrics["total_return"] * 100,
                risk_scalar,
                self.regime_scalar_cap,
                time.time() - t0_period,
            )

            period_num += 1
            start_idx += self.test

        return self.get_results()

    def get_results(self) -> dict:
        if not self.results:
            return self._empty_result()

        df = pd.DataFrame(self.results)

        def cagr(returns_list: list[float]) -> float:
            if not returns_list:
                return 0.0
            cumulative = 1.0
            for r in returns_list:
                cumulative *= 1 + float(r)
            periods_per_year = TRADING_DAYS / self.test
            years = len(returns_list) / periods_per_year
            if years <= 0:
                return 0.0
            return float(cumulative ** (1 / years) - 1)

        hrp_cagr = cagr(df["hrp_return"].tolist())
        bench_cagr = cagr(df["eq_return"].tolist())

        hrp_sharpe = float(df["hrp_sharpe"].mean())
        bench_sharpe = float(df["eq_sharpe"].mean())

        hrp_dd = float(df["hrp_drawdown"].min())
        bench_dd = float(df["eq_drawdown"].min())

        eq_curve_hrp = []
        eq_curve_bench = []
        cum_hrp = 1.0
        cum_bench = 1.0
        for _, row in df.iterrows():
            cum_hrp *= 1 + float(row["hrp_return"])
            cum_bench *= 1 + float(row["eq_return"])
            eq_curve_hrp.append({"date": row["test_end"], "value": round(cum_hrp - 1, 6)})
            eq_curve_bench.append({"date": row["test_end"], "value": round(cum_bench - 1, 6)})

        bear_periods = df[df["regime_bear"] == True]
        if len(bear_periods) > 0:
            regime_timing = float(bear_periods["hrp_return"].mean() - bear_periods["eq_return"].mean())
        else:
            regime_timing = 0.0

        result = {
            "hrp_strategy_cagr": round(hrp_cagr, 4),
            "benchmark_cagr": round(bench_cagr, 4),
            "hrp_sharpe": round(hrp_sharpe, 4),
            "benchmark_sharpe": round(bench_sharpe, 4),
            "hrp_max_drawdown": round(hrp_dd, 4),
            "benchmark_max_drawdown": round(bench_dd, 4),
            "regime_timing_value": round(regime_timing, 4),
            "equity_curve_hrp": eq_curve_hrp,
            "equity_curve_benchmark": eq_curve_bench,
            "n_periods": len(df),
            "window_days": self.train,
            "test_days": self.test,
            "periods_detail": df.to_dict(orient="records"),
        }

        # Rolling Sharpe per period (for chart)
        rolling_sharpe = [
            {
                "period": p["period"],
                "date": p["test_end"],
                "hrp": round(p["hrp_sharpe"], 4),
                "benchmark": round(p["eq_sharpe"], 4),
            }
            for p in self.results
        ]

        # Validity note
        n = len(df)
        if n < 8:
            validity_note = (
                f"CAUTION: Only {n} periods — "
                f"results are not statistically reliable. "
                f"Run with --price-period 5y for ~15 periods."
            )
            self.logger.warning("Backtest: %s", validity_note)
        elif n < 12:
            validity_note = f"ACCEPTABLE: {n} periods — directional validity only."
        else:
            validity_note = f"VALID: {n} periods — statistically meaningful."

        # Risk-adjusted framing
        dd_improvement = 0.0
        if bench_dd != 0:
            dd_improvement = (abs(bench_dd) - abs(hrp_dd)) / abs(bench_dd)

        result.update(
            {
                "rolling_sharpe": rolling_sharpe,
                "validity_note": validity_note,
                "drawdown_reduction_pct": round(dd_improvement, 4),
                "regime_scalar_cap": self.regime_scalar_cap,
                "interpretation": (
                    "HRP+Regime optimizes for capital "
                    "preservation in Bear regimes. "
                    "Lower CAGR vs equal-weight is expected "
                    "when risk_scalar is high. "
                    f"Key metric: {dd_improvement:.1%} less "
                    "drawdown vs benchmark."
                ),
            }
        )

        self.logger.info(
            "Backtest complete: %d periods | HRP CAGR=%.2f%% vs EQ=%.2f%% | HRP Sharpe=%.3f vs EQ=%.3f | Regime timing value=%.4f",
            len(df),
            hrp_cagr * 100,
            bench_cagr * 100,
            hrp_sharpe,
            bench_sharpe,
            regime_timing,
        )
        return result

    def _empty_result(self) -> dict:
        return {
            "hrp_strategy_cagr": 0.0,
            "benchmark_cagr": 0.0,
            "hrp_sharpe": 0.0,
            "benchmark_sharpe": 0.0,
            "hrp_max_drawdown": 0.0,
            "benchmark_max_drawdown": 0.0,
            "regime_timing_value": 0.0,
            "equity_curve_hrp": [],
            "equity_curve_benchmark": [],
            "n_periods": 0,
            "window_days": self.train,
            "test_days": self.test,
            "periods_detail": [],
        }

    def get_committee_input(self) -> dict:
        if not self.results:
            return self._empty_result()
        return self.get_results()
