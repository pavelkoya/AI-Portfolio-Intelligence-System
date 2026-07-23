from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class DataValidator:
    """Validates and cleans data before sending it to the quant engine."""

    PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]

    def _error(self, messages: List[str], msg: str) -> None:
        messages.append(msg)
        logger.error(msg)

    def validate_prices(self, df: DataFrame, ticker: str) -> Tuple[bool, List[str]]:
        """
        Validate historical OHLCV prices for a ticker.

        Returns:
            (True, []) if valid, else (False, [error messages])
        """
        logger.info("validate_prices: validating %s", ticker)
        errors: List[str] = []

        if df is None or df.empty:
            errors.append(f"{ticker}: prices dataframe is empty")
            logger.error("%s: prices dataframe is empty", ticker)
            return False, errors

        missing_cols = [c for c in self.PRICE_COLUMNS if c not in df.columns]
        if missing_cols:
            errors.append(f"{ticker}: missing required columns: {missing_cols}")
            logger.error("%s: missing required columns: %s", ticker, missing_cols)
            return False, errors

        if "date" not in df.columns:
            errors.append(f"{ticker}: missing required column: date")
            logger.error("%s: missing required column date", ticker)
            return False, errors

        numeric_df = df.copy()
        for c in self.PRICE_COLUMNS:
            numeric_df[c] = pd.to_numeric(numeric_df[c], errors="coerce")

        # Null value ratio across date + required columns
        subset = numeric_df[["date"] + self.PRICE_COLUMNS]
        total_cells = len(subset) * subset.shape[1] if len(subset) else 0
        null_cells = int(subset.isnull().sum().sum()) if total_cells else 0
        null_ratio = (null_cells / total_cells) if total_cells else 1.0
        if null_ratio >= 0.10:
            errors.append(
                f"{ticker}: too many null values ({null_ratio:.2%} >= 10%)"
            )
            logger.error("%s: too many null values (%.2f%%)", ticker, null_ratio * 100)
            return False, errors

        # Coerce date & check monotonic increasing
        dates = pd.to_datetime(df["date"], errors="coerce")
        if dates.isnull().any():
            errors.append(f"{ticker}: invalid/unparseable dates present")
            logger.error("%s: invalid/unparseable dates present", ticker)
            return False, errors
        # Monotonic increasing allows equal dates; cleaning should remove duplicates separately.
        if not dates.is_monotonic_increasing:
            errors.append(f"{ticker}: dates are not monotonically increasing")
            logger.error("%s: dates are not monotonically increasing", ticker)
            return False, errors

        # Check negative prices
        price_df = df[self.PRICE_COLUMNS].copy()
        for c in self.PRICE_COLUMNS:
            price_df[c] = pd.to_numeric(price_df[c], errors="coerce")
        if (price_df[["open", "high", "low", "close"]] < 0).any().any():
            errors.append(f"{ticker}: negative OHLC prices detected")
            logger.error("%s: negative OHLC prices detected", ticker)
            return False, errors
        if (price_df["volume"] < 0).any():
            errors.append(f"{ticker}: negative volume detected")
            logger.error("%s: negative volume detected", ticker)
            return False, errors

        # Enough rows
        if len(df) < 252:
            errors.append(f"{ticker}: not enough rows ({len(df)} < 252)")
            logger.error("%s: not enough rows (%d < 252)", ticker, len(df))
            return False, errors

        return True, []

    def validate_portfolio(
        self, positions: List[Dict[str, Any]]
    ) -> Tuple[bool, List[str]]:
        """Validate portfolio positions before passing to quant engine."""
        logger.info("validate_portfolio: validating portfolio (%d positions)", len(positions))
        errors: List[str] = []

        if not positions:
            errors.append("portfolio is empty")
            logger.error("validate_portfolio: portfolio is empty")
            return False, errors

        invalid_tickers: List[str] = []
        for idx, pos in enumerate(positions):
            if not isinstance(pos, dict):
                errors.append(f"position[{idx}]: position must be a dict")
                logger.error("position[%d]: position must be a dict", idx)
                continue

            for key in ("ticker", "quantity", "average_buy_price"):
                if key not in pos:
                    errors.append(f"position[{idx}]: missing required key: {key}")
                    logger.error("position[%d]: missing required key: %s", idx, key)

            ticker = pos.get("ticker")
            quantity = pos.get("quantity")
            avg_buy = pos.get("average_buy_price")

            if ticker is None or not isinstance(ticker, str):
                errors.append(f"position[{idx}]: ticker must be a string")
                logger.error("position[%d]: ticker must be a string", idx)
            else:
                normalized = ticker.upper().strip()
                # Spec: uppercase, 1-5 chars
                if (
                    normalized != ticker
                    or not normalized.isupper()
                    or not (1 <= len(normalized) <= 5)
                    or not normalized.isalnum()
                ):
                    errors.append(
                        f"position[{idx}]: invalid ticker format: {ticker} (expected uppercase 1-5 chars)"
                    )
                    invalid_tickers.append(ticker)

            try:
                qty_f = float(quantity)
            except Exception:
                errors.append(f"position[{idx}]: quantity is not numeric: {quantity}")
                logger.error("position[%d]: quantity is not numeric: %s", idx, quantity)
                continue

            try:
                avg_f = float(avg_buy)
            except Exception:
                errors.append(
                    f"position[{idx}]: average_buy_price is not numeric: {avg_buy}"
                )
                logger.error(
                    "position[%d]: average_buy_price is not numeric: %s",
                    idx,
                    avg_buy,
                )
                continue

            if qty_f < 0:
                errors.append(f"position[{idx}]: negative quantity not allowed")
                logger.error("position[%d]: negative quantity not allowed", idx)

        if errors:
            return False, errors
        return True, []

    def clean_prices(self, df: DataFrame) -> DataFrame:
        """Clean price data by forward-filling short gaps and removing bad rows."""
        logger.info("clean_prices: cleaning prices dataframe")
        if df is None or df.empty:
            return pd.DataFrame(columns=["date", *self.PRICE_COLUMNS])

        cleaned = df.copy()

        if "date" not in cleaned.columns:
            raise ValueError("clean_prices: missing required column: date")

        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
        cleaned = cleaned.dropna(subset=["date"])

        for c in self.PRICE_COLUMNS:
            if c in cleaned.columns:
                cleaned[c] = pd.to_numeric(cleaned[c], errors="coerce")

        # Ensure required columns exist for ffill
        for c in self.PRICE_COLUMNS:
            if c not in cleaned.columns:
                cleaned[c] = pd.NA

        # Sort before ffill so we fill forward in time.
        cleaned = cleaned.sort_values("date", ascending=True)

        cleaned[self.PRICE_COLUMNS] = cleaned[self.PRICE_COLUMNS].ffill(limit=3)

        # Drop rows where close is still missing after fill
        cleaned = cleaned.dropna(subset=["close"])

        # Remove duplicate dates (keep last)
        cleaned = cleaned.drop_duplicates(subset=["date"], keep="last")

        cleaned = cleaned.sort_values("date", ascending=True).reset_index(drop=True)
        return cleaned[["date", *self.PRICE_COLUMNS]]

    def validate_all(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean the full data dict from :meth:`data.data_loader.DataLoader.get_all_data`.
        """
        logger.info("validate_all: starting")

        portfolio = data.get("portfolio") or []
        prices: Dict[str, DataFrame] = data.get("prices") or {}
        news: Dict[str, List[str]] = data.get("news") or {}
        benchmark_df = data.get("benchmark")

        # Validate/clean portfolio
        portfolio_valid, portfolio_errors = self.validate_portfolio(portfolio)
        if not portfolio_valid:
            # Only critical failure is empty portfolio.
            if not portfolio:
                logger.error("validate_all: empty portfolio (critical)")
                raise ValueError("empty portfolio")

            logger.warning("validate_all: portfolio invalid; attempting to clean")
            logger.warning("validate_all: portfolio errors: %s", portfolio_errors)

            cleaned_positions: List[Dict[str, Any]] = []
            for pos in portfolio:
                if not isinstance(pos, dict):
                    continue
                ticker = pos.get("ticker")
                quantity = pos.get("quantity")
                avg_buy = pos.get("average_buy_price")

                if not isinstance(ticker, str):
                    continue
                ticker_norm = ticker.upper().strip()
                if (
                    not ticker_norm.isupper()
                    or not (1 <= len(ticker_norm) <= 5)
                    or not ticker_norm.isalnum()
                ):
                    continue
                try:
                    qty_f = float(quantity)
                    avg_f = float(avg_buy)
                except Exception:
                    continue
                if qty_f < 0:
                    continue

                pos_clean = dict(pos)
                pos_clean["ticker"] = ticker_norm
                pos_clean["quantity"] = qty_f
                pos_clean["average_buy_price"] = avg_f
                cleaned_positions.append(pos_clean)

            if not cleaned_positions:
                logger.error("validate_all: no valid portfolio positions after cleaning (critical)")
                raise ValueError("empty portfolio")

            portfolio = cleaned_positions

        # Validate/clean prices per portfolio ticker
        tickers = [p.get("ticker") for p in portfolio if p.get("ticker")]
        tickers = [str(t).upper().strip() for t in tickers if t]
        unique_tickers = list(dict.fromkeys(tickers))

        cleaned_prices: Dict[str, DataFrame] = {}
        cleaned_news: Dict[str, List[str]] = {}
        invalid_tickers: List[str] = []

        for ticker in unique_tickers:
            df = prices.get(ticker)
            if df is None or (isinstance(df, DataFrame) and df.empty):
                invalid_tickers.append(ticker)
                logger.warning("validate_all: %s: missing/empty price data", ticker)
                continue

            try:
                cleaned_df = self.clean_prices(df)
            except Exception as e:
                invalid_tickers.append(ticker)
                logger.warning("validate_all: %s: failed cleaning: %s", ticker, str(e))
                continue

            valid, errors = self.validate_prices(cleaned_df, ticker)
            if not valid:
                invalid_tickers.append(ticker)
                logger.warning(
                    "validate_all: %s: price validation failed: %s", ticker, errors
                )
                continue

            cleaned_prices[ticker] = cleaned_df
            cleaned_news[ticker] = news.get(ticker, []) or []

        if not cleaned_prices:
            # Critical failure: all tickers invalid.
            logger.error("validate_all: all tickers invalid (critical)")
            raise ValueError("all tickers invalid")

        # Clean benchmark prices (best-effort; no critical raise)
        cleaned_benchmark = pd.DataFrame()
        if benchmark_df is not None and isinstance(benchmark_df, DataFrame) and not benchmark_df.empty:
            try:
                cleaned_benchmark = self.clean_prices(benchmark_df)
                valid_bm, bm_errors = self.validate_prices(cleaned_benchmark, "BENCHMARK")
                if not valid_bm:
                    logger.warning("validate_all: benchmark validation failed: %s", bm_errors)
                    cleaned_benchmark = pd.DataFrame()
            except Exception as e:
                logger.warning("validate_all: benchmark cleaning failed: %s", str(e))
                cleaned_benchmark = pd.DataFrame()
        else:
            logger.warning("validate_all: benchmark dataframe missing/empty")

        # Remove invalid tickers from news/portfolio ticker list
        valid_ticker_set = set(cleaned_prices.keys())
        portfolio = [p for p in portfolio if p.get("ticker") in valid_ticker_set]

        logger.info(
            "validate_all: finished (valid_tickers=%d, invalid_tickers=%d)",
            len(cleaned_prices),
            len(invalid_tickers),
        )
        return {
            "portfolio": portfolio,
            "prices": cleaned_prices,
            "news": cleaned_news,
            "benchmark": cleaned_benchmark,
        }
