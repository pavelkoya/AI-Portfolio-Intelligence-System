from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from config.broker_maps import (
    BROKER_MAPS,
    detect_broker,
    load_user_maps,
    normalize_ticker,
)


class CSVImporter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.raw_df = None
        self.broker = None
        self.warnings = []
        self.column_map = {}

    def load_file(self, filepath: str) -> pd.DataFrame:
        """
        Load CSV or Excel file.
        """
        csv_err = None
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            csv_err = e
            try:
                df = pd.read_excel(filepath)
            except Exception as ex:
                raise ValueError(
                    f"Could not read file as CSV or Excel: {filepath}"
                ) from ex

        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(how="all")
        self.raw_df = df

        # Detect Freedom Broker Excel structure:
        # Header row has "Ticker" in column index 1
        # and section rows have text in column 0
        if "Ticker" in df.columns or (
            len(df) > 0 and df.iloc[0].astype(str).str.contains("Ticker").any()
        ):
            # Re-read with correct header
            # Try reading with row 0 as header
            pass  # pandas already handled this

        # Filter out section header rows:
        # These have NaN ticker and non-NaN first col
        # OR have "Instruments in" in any column
        if self.raw_df is not None:
            ticker_col = None
            for col in self.raw_df.columns:
                if str(col).strip() == "Ticker":
                    ticker_col = col
                    break
            if ticker_col:
                self.raw_df = self.raw_df[
                    self.raw_df[ticker_col].notna()
                    & ~self.raw_df[ticker_col]
                    .astype(str)
                    .str.contains("Instruments|NaN", na=True)
                ].reset_index(drop=True)

        if csv_err is not None:
            self.logger.info("load_file: CSV parse failed, loaded as Excel: %s", filepath)
        else:
            self.logger.info("load_file: loaded CSV: %s", filepath)
        return self.raw_df

    def load_two_files(self, path1: str, path2: str) -> pd.DataFrame:
        """
        Load two files and merge them.
        """
        df1 = self.load_file(path1).copy()
        df2 = self.load_file(path2).copy()
        merged = pd.concat([df1, df2], ignore_index=True)

        user_maps = load_user_maps()
        all_maps = {**BROKER_MAPS, **user_maps}
        broker = detect_broker(merged.columns.tolist(), all_maps)
        mapping = all_maps.get(broker, {})

        ticker_col = None
        value_col = None

        for v in mapping.get("ticker", []):
            if v in merged.columns:
                ticker_col = v
                break
        for v in mapping.get("market_value", []):
            if v in merged.columns:
                value_col = v
                break

        if ticker_col and value_col:
            tmp = merged.copy()
            tmp[value_col] = tmp[value_col].apply(self._to_float)
            tmp = tmp.sort_values(value_col, ascending=False)
            merged = tmp.drop_duplicates(subset=[ticker_col], keep="first")
        elif ticker_col:
            merged = merged.drop_duplicates(subset=[ticker_col], keep="first")

        merged = merged.reset_index(drop=True)
        self.raw_df = merged
        return merged

    def detect_and_map(self) -> Optional[dict]:
        """
        Auto-detect broker and build column_map.
        Returns column_map dict or None if not detected.
        """
        if self.raw_df is None:
            raise ValueError("No file loaded. Call load_file() first.")

        cols = self.raw_df.columns.tolist()
        user_maps = load_user_maps()
        all_maps = {**BROKER_MAPS, **user_maps}

        broker = detect_broker(cols, all_maps)
        if not broker:
            self.broker = None
            self.column_map = {}
            return None

        self.broker = broker
        mapping = all_maps.get(broker, {})

        self.column_map = {}
        for field, variants in mapping.items():
            for variant in variants:
                if variant in self.raw_df.columns:
                    self.column_map[field] = variant
                    break

        return self.column_map

    def to_standard_schema(self) -> List[dict]:
        """
        Convert raw_df to standard portfolio list.
        """
        if not self.column_map:
            raise ValueError("No column map available. Call detect_and_map() first.")
        if self.raw_df is None:
            raise ValueError("No file loaded. Call load_file() first.")

        positions: List[dict] = []

        for _, row in self.raw_df.iterrows():
            raw_tick = row.get(self.column_map.get("ticker", ""), "")
            if pd.isna(raw_tick) or str(raw_tick).strip() == "":
                continue

            ticker, warn = normalize_ticker(str(raw_tick))
            if warn:
                self.warnings.append(warn)

            shares = self._to_float(row.get(self.column_map.get("shares", ""), 0))
            avg_price = self._to_float(row.get(self.column_map.get("avg_price", ""), 0))
            market_value = self._to_float(row.get(self.column_map.get("market_value", ""), 0))
            weight_pct = self._to_float(row.get(self.column_map.get("weight_pct", ""), 0))
            pnl_pct = self._to_float(
                row.get(self.column_map.get("unrealized_pnl_pct", ""), 0)
            )
            current_price_col = self.column_map.get("current_price", "")
            current_price_raw = row.get(current_price_col, 0)
            try:
                cp_text = str(current_price_raw).replace(",", "")
                current_price = float(cp_text.split()[0] or 0)
            except (ValueError, TypeError):
                current_price = 0.0

            positions.append(
                {
                    "ticker": ticker,
                    "quantity": shares,
                    "average_buy_price": avg_price,
                    "current_price": current_price,
                    "market_value": market_value,
                    "weight_pct": weight_pct,
                    "unrealized_pnl_pct": pnl_pct,
                }
            )

        tickers_to_fetch = [p["ticker"] for p in positions if p.get("ticker")]
        latest: Dict[str, float] = {}
        if tickers_to_fetch:
            try:
                data = yf.download(
                    tickers_to_fetch,
                    period="1d",
                    progress=False,
                    auto_adjust=False,
                )
                if isinstance(data.columns, pd.MultiIndex):
                    close = data["Close"]
                else:
                    close = data[["Close"]]
                    if len(tickers_to_fetch) == 1:
                        close.columns = [tickers_to_fetch[0]]

                if not close.empty:
                    latest = close.iloc[-1].to_dict()
            except Exception:
                latest = {}
                self.warnings.append("Could not fetch current prices from yfinance")

        for pos in positions:
            if not pos.get("current_price"):
                current_price = latest.get(pos["ticker"], pos["average_buy_price"])
                pos["current_price"] = float(current_price or 0)

            # Auto-calculate market_value from shares × price
            # if market_value is missing but both are present
            if (
                not pos["market_value"]
                and pos["quantity"]
                and pos["current_price"]
            ):
                pos["market_value"] = pos["quantity"] * pos["current_price"]

            # Auto-calculate unrealized_pnl
            if pos["average_buy_price"] > 0 and pos["current_price"] > 0:
                pos["unrealized_pnl_pct"] = (
                    (pos["current_price"] - pos["average_buy_price"])
                    / pos["average_buy_price"]
                    * 100
                )
                pos["unrealized_pnl"] = (
                    (pos["current_price"] - pos["average_buy_price"]) * pos["quantity"]
                )
            else:
                pos["unrealized_pnl_pct"] = 0.0
                pos["unrealized_pnl"] = 0.0

        known_suffixes = (
            ".US",
            ".EU",
            ".DE",
            ".L",
            ".PA",
            ".AS",
            ".MI",
            ".MC",
            ".HK",
            ".T",
            ".SS",
            ".SZ",
            ".ME",
            ".TO",
            ".AX",
        )
        for pos in positions:
            if pos.get("current_price", 0) != 0:
                continue
            ticker = str(pos.get("ticker", "")).upper()
            if not ticker or ticker.endswith(known_suffixes):
                continue
            fallback_ticker = f"{ticker}.DE"
            try:
                fb = yf.download(
                    fallback_ticker,
                    period="1d",
                    progress=False,
                    auto_adjust=False,
                )
                if not fb.empty and "Close" in fb.columns:
                    fallback_price = self._to_float(fb["Close"].iloc[-1])
                    if fallback_price > 0:
                        pos["current_price"] = fallback_price
                        if not pos["market_value"] and pos["quantity"]:
                            pos["market_value"] = pos["quantity"] * pos["current_price"]
                        if pos["average_buy_price"] > 0 and pos["current_price"] > 0:
                            pos["unrealized_pnl_pct"] = (
                                (pos["current_price"] - pos["average_buy_price"])
                                / pos["average_buy_price"]
                                * 100
                            )
                            pos["unrealized_pnl"] = (
                                (pos["current_price"] - pos["average_buy_price"])
                                * pos["quantity"]
                            )
                        self.warnings.append(f"Used .DE fallback for {ticker}")
            except Exception:
                continue

        return positions

    def apply_manual_map(self, column_map: dict) -> None:
        """
        Apply a manually provided column mapping.
        column_map: {standard_field: raw_column_name}
        Saves mapping to user_broker_maps.json.
        """
        self.column_map = column_map or {}
        try:
            from config.broker_maps import save_user_map

            broker_name = self.broker or "manual"
            save_user_map(broker_name, self.column_map)
        except Exception as e:
            self.logger.warning("apply_manual_map: could not save user map: %s", e)

    @staticmethod
    def _to_float(value) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float, np.integer, np.floating)):
            if pd.isna(value):
                return 0.0
            return float(value)

        s = str(value).strip()
        if s == "" or s.lower() in {"nan", "none", "-"}:
            return 0.0
        s = s.replace("%", "").replace(" ", "")
        if "," in s and "." not in s:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
        try:
            return float(s)
        except Exception:
            return 0.0
