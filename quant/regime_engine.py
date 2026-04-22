from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm

from config.settings import TRADING_DAYS
from data.database import DatabaseManager


class RegimeEngine:
    def __init__(self, benchmark: pd.DataFrame):
        # benchmark: SPY DataFrame from Phase 1 data dict
        # columns: [date, open, high, low, close, volume]
        self.benchmark = benchmark
        self.model = None
        self.regime_map: dict[int, str] = {}  # maps HMM state → label
        self.logger = logging.getLogger(__name__)
        self.features: np.ndarray | None = None  # will store feature matrix

        self._feature_dates: pd.DatetimeIndex | None = None
        self._spy_returns_aligned: pd.Series | None = None
        self._state_means: dict[int, float] = {}

    def _fetch_vix(self) -> pd.Series:
        try:
            vix_raw = yf.download(
                "^VIX",
                period="3y",
                interval="1d",
                progress=False,
                auto_adjust=False,
            )
            if vix_raw is None or getattr(vix_raw, "empty", True):
                self.logger.warning("RegimeEngine: VIX download returned empty")
                return None

            # Flatten MultiIndex columns (yfinance sometimes returns MultiIndex)
            if isinstance(vix_raw.columns, pd.MultiIndex):
                vix_raw.columns = [col[0] for col in vix_raw.columns]

            if "Close" not in vix_raw.columns:
                self.logger.warning("RegimeEngine: VIX data missing Close column")
                return None

            close = vix_raw["Close"]
            close = pd.to_numeric(close, errors="coerce").dropna()
            close.index = pd.to_datetime(close.index, errors="coerce").normalize()
            close = close.dropna()
            close.name = "vix_close"
            return close
        except Exception:
            self.logger.warning("RegimeEngine: failed to fetch VIX", exc_info=True)
            return None

    def _prepare_features(self) -> np.ndarray:
        spy_df = self.benchmark.copy()
        if spy_df is None or getattr(spy_df, "empty", True):
            self.logger.error("RegimeEngine: empty benchmark (SPY) dataframe")
            raise ValueError("SPY benchmark data is empty")

        if "date" not in spy_df.columns or "close" not in spy_df.columns:
            self.logger.error("RegimeEngine: SPY data missing required columns")
            raise ValueError("SPY benchmark data missing 'date' or 'close'")

        spy_df["date"] = pd.to_datetime(spy_df["date"]).dt.normalize()
        spy_df["close"] = pd.to_numeric(spy_df["close"], errors="coerce")
        spy_df = spy_df.dropna(subset=["date", "close"]).sort_values("date")

        if len(spy_df) < 504:
            self.logger.error(
                "RegimeEngine: insufficient SPY history (%d rows; need >= 504)",
                len(spy_df),
            )
            raise ValueError("Insufficient SPY history for regime detection")

        spy_df = spy_df.set_index("date")
        spy_returns = spy_df["close"].pct_change().dropna()
        spy_returns.name = "spy_return"

        vix = self._fetch_vix()
        if vix is not None:
            aligned = pd.concat([spy_returns, vix], axis=1, join="inner").dropna()
            vix_aligned = aligned["vix_close"] / 100.0
            spy_ret_aligned = aligned["spy_return"]
            X = np.column_stack([spy_ret_aligned.values, vix_aligned.values])
            dates = aligned.index
        else:
            self.logger.warning("RegimeEngine: VIX unavailable; using SPY returns only")
            spy_ret_aligned = spy_returns
            X = spy_ret_aligned.values.reshape(-1, 1)
            dates = spy_ret_aligned.index

        lookback = 3 * TRADING_DAYS
        if len(dates) > lookback:
            X = X[-lookback:]
            dates = dates[-lookback:]
            spy_ret_aligned = spy_ret_aligned.iloc[-lookback:]

        self.features = X
        self._feature_dates = pd.DatetimeIndex(dates)
        self._spy_returns_aligned = spy_ret_aligned
        return X

    def _map_states_to_labels(self, states: np.ndarray, returns: pd.Series) -> dict:
        means: dict[int, float] = {}
        for s in (0, 1, 2):
            mask = states == s
            if mask.sum() == 0:
                means[s] = float("nan")
            else:
                means[s] = float(np.nanmean(returns.values[mask]))

        # Sort states by mean return ascending
        ordered = sorted(means.keys(), key=lambda k: (np.nan_to_num(means[k], nan=-1e9)))
        mapping = {ordered[0]: "Bear", ordered[1]: "Neutral", ordered[2]: "Bull"}
        self._state_means = means
        return mapping

    def fit(self) -> None:
        self._prepare_features()
        self.model = hmm.GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        self.model.fit(self.features)

        states = self.model.predict(self.features)
        returns = self._spy_returns_aligned
        self.regime_map = self._map_states_to_labels(states, returns)

        # Log state means in label terms for readability
        label_means = {self.regime_map[s]: round(float(m), 6) for s, m in self._state_means.items()}
        self.logger.info("HMM fitted. State means: %s", label_means)

    def predict_current(self) -> dict:
        if self.model is None or self.features is None or not self.regime_map:
            self.fit()

        probs = self.model.predict_proba(self.features)[-1]

        # Identify crash/bear/bull via mean returns
        ordered = sorted(
            self._state_means.keys(),
            key=lambda k: (np.nan_to_num(self._state_means[k], nan=-1e9)),
        )
        crash_state = ordered[0]  # lowest mean (worst)
        bear_state = ordered[1]  # second lowest mean

        crash_prob = float(probs[crash_state])
        bear_prob = float(probs[bear_state])
        risk_scalar = (bear_prob * 0.5) + (crash_prob * 1.0)
        risk_scalar = min(max(risk_scalar, 0.0), 1.0)

        # Map probabilities to labels using regime_map
        label_probs: dict[str, float] = {"Bull": 0.0, "Neutral": 0.0, "Bear": 0.0}
        for state_idx, p in enumerate(probs):
            label = self.regime_map.get(int(state_idx), "Neutral")
            label_probs[label] = label_probs.get(label, 0.0) + float(p)

        dominant_regime = max(label_probs.items(), key=lambda kv: kv[1])[0]
        return {
            "Bull": round(float(label_probs.get("Bull", 0.0)), 3),
            "Neutral": round(float(label_probs.get("Neutral", 0.0)), 3),
            "Bear": round(float(label_probs.get("Bear", 0.0)), 3),
            "dominant_regime": dominant_regime,
            "risk_scalar": round(float(risk_scalar), 3),
        }

    def backcast_validation(self) -> dict:
        if self.model is None or self.features is None or not self.regime_map:
            self.fit()

        states = self.model.predict(self.features)
        labels = np.array([self.regime_map.get(int(s), "Unknown") for s in states], dtype=object)
        dates = self._feature_dates

        def dominant_between(start: str, end: str) -> str:
            start_dt = pd.to_datetime(start).normalize()
            end_dt = pd.to_datetime(end).normalize()
            mask = (dates >= start_dt) & (dates <= end_dt)
            if mask.sum() == 0:
                return "Unknown"
            vals = pd.Series(labels[mask]).value_counts()
            return str(vals.index[0]) if not vals.empty else "Unknown"

        march_2020 = dominant_between("2020-02-20", "2020-04-01")
        year_2022 = dominant_between("2022-01-01", "2022-12-31")

        out = {
            "march_2020_dominant_regime": march_2020,
            "year_2022_dominant_regime": year_2022,
            "validation_note": "Expected Bear/Crash for both periods",
        }
        self.logger.info("Regime backcast validation: %s", out)
        return out

    def run(self) -> dict:
        self.fit()
        current = self.predict_current()
        validation = self.backcast_validation()
        self.logger.info(
            "RegimeEngine.run: dominant_regime=%s risk_scalar=%.3f",
            current.get("dominant_regime"),
            float(current.get("risk_scalar") or 0.0),
        )
        return {"current": current, "validation": validation}
