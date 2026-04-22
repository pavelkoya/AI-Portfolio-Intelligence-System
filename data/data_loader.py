from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from pandas import DataFrame

from config.settings import BENCHMARK_TICKER, ROBINHOOD_PASSWORD, ROBINHOOD_USERNAME
from data.database import DatabaseManager

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class DataLoader:
    """
    Fetches portfolio snapshots (Robinhood) and historical prices/news (Robinhood + yfinance).

    Persistence is handled by :class:`data.database.DatabaseManager`.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()
        self.logger = logger
        try:
            self.db.init_db()
        except Exception:
            # Don't fail hard for init errors; API methods will surface if needed.
            logger.exception("DataLoader: failed to init database")

        self._robinhood_logged_in = False

    def login_robinhood(self) -> bool:
        """Log in to Robinhood using configured credentials."""
        logger.info("login_robinhood: starting")
        try:
            try:
                from robin_stocks.robinhood.authentication import login as rh_login
            except ImportError as e:
                logger.exception("login_robinhood: robin_stocks not available")
                return False

            username = ROBINHOOD_USERNAME or ""
            password = ROBINHOOD_PASSWORD or ""

            if not username or not password:
                logger.error(
                    "login_robinhood: missing ROBINHOOD_USERNAME/ROBINHOOD_PASSWORD in environment"
                )
                return False

            result = rh_login(username=username, password=password)
            success = result is not None
            self._robinhood_logged_in = success

            if success:
                logger.info("login_robinhood: success")
            else:
                logger.warning("login_robinhood: failed (empty response)")
            return success
        except (KeyboardInterrupt, EOFError) as e:
            # Robinhood login can prompt for MFA (input()). In headless runs this can fail.
            logger.error("login_robinhood: aborted during MFA prompt: %s", str(e))
            self._robinhood_logged_in = False
            return False
        except Exception as e:
            logger.exception("login_robinhood: exception while logging in: %s", str(e))
            self._robinhood_logged_in = False
            return False
        finally:
            logger.info(
                "login_robinhood: finished (logged_in=%s)", self._robinhood_logged_in
            )

    def _safe_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            # robin_stocks often returns strings for numeric fields
            return float(str(value).strip())
        except Exception:
            return None

    def get_portfolio(self) -> List[Dict[str, Any]]:
        """Fetch portfolio holdings from Robinhood and persist a snapshot."""
        logger.info("get_portfolio: starting")
        if not self._robinhood_logged_in:
            logger.warning(
                "get_portfolio: Robinhood not logged in; returning empty portfolio"
            )
            return []

        try:
            from robin_stocks.robinhood import account as rh_account

            holdings = rh_account.build_holdings()
            if not holdings:
                logger.warning("get_portfolio: no holdings returned")
                return []

            positions: List[Dict[str, Any]] = []
            for ticker, pos in holdings.items():
                if not ticker or not isinstance(pos, dict):
                    continue

                raw_type = str(pos.get("type", "") or "").lower()
                equity_type = "crypto" if "crypto" in raw_type else "stock"

                positions.append(
                    {
                        "ticker": str(ticker).upper().strip(),
                        "quantity": self._safe_float(pos.get("quantity")),
                        "average_buy_price": self._safe_float(pos.get("average_buy_price")),
                        "current_price": self._safe_float(pos.get("price")),
                        "market_value": self._safe_float(pos.get("equity")),
                        "unrealized_pnl": self._safe_float(pos.get("equity_change")),
                        "unrealized_pnl_pct": self._safe_float(pos.get("percent_change")),
                        "equity_type": equity_type,
                    }
                )

            if positions:
                try:
                    self.db.save_portfolio_snapshot(positions)
                    logger.info(
                        "get_portfolio: saved portfolio snapshot (%d positions)",
                        len(positions),
                    )
                except Exception as e:
                    logger.exception(
                        "get_portfolio: failed saving snapshot to DB: %s", str(e)
                    )
            else:
                logger.warning("get_portfolio: holdings parsed to empty positions list")

            return positions
        except Exception as e:
            logger.exception("get_portfolio: exception while building holdings: %s", str(e))
            return []
        finally:
            logger.info("get_portfolio: finished")

    def _normalize_yfinance_ohlcv(self, df: DataFrame) -> DataFrame:
        """Convert yfinance output to DB/validator-friendly column names."""
        if df is None or df.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        out = df.copy()

        def _flatten_yf_multiindex_column(c: Any) -> Optional[str]:
            """Map MultiIndex levels to OHLCV names; None means drop (e.g. Adj Close)."""
            if isinstance(c, tuple):
                c0 = str(c[0])
                c1 = str(c[1]) if len(c) > 1 else None
                if c0 == "Adj Close":
                    return None
                if (
                    c0 == "Price"
                    and c1 is not None
                    and c1 in ("Open", "High", "Low", "Close", "Volume")
                ):
                    return c1
                return c0
            return str(c)

        # yfinance sometimes returns MultiIndex columns even for single tickers.
        if isinstance(out.columns, pd.MultiIndex):
            keep_cols: List[Any] = []
            new_names: List[str] = []
            for c in out.columns:
                name = _flatten_yf_multiindex_column(c)
                if name is None:
                    continue
                keep_cols.append(c)
                new_names.append(name)
            out = out[keep_cols].copy()
            out.columns = new_names
        elif "Adj Close" in out.columns:
            out = out.drop(columns=["Adj Close"])

        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        out = out.rename(columns=rename_map)

        allowed = ["open", "high", "low", "close", "volume"]
        out = out[[c for c in out.columns if c in allowed]]

        required = ["open", "high", "low", "close", "volume"]
        if not all(c in out.columns for c in required):
            missing = [c for c in required if c not in out.columns]
            raise ValueError(f"yfinance output missing OHLCV columns: {missing}")

        out = out.reset_index()
        out = out.rename(columns={"Date": "date", "index": "date"})
        if "date" not in out.columns and len(out.columns) > 0:
            first = out.columns[0]
            if first not in required:
                out = out.rename(columns={first: "date"})
        if "date" not in out.columns:
            raise ValueError("yfinance output missing date column after reset_index")

        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"])

        for c in required:
            out[c] = pd.to_numeric(out[c], errors="coerce")

        return out[["date", "open", "high", "low", "close", "volume"]]

    def get_historical_prices(
        self, tickers: List[str], period: str = "2y"
    ) -> Dict[str, DataFrame]:
        """
        Download historical OHLCV for tickers + benchmark (SPY).

        Returns:
            dict mapping symbol -> cleaned DataFrame with columns:
            date, open, high, low, close, volume
        """
        logger.info(
            "get_historical_prices: starting (tickers=%s, period=%s)",
            tickers,
            period,
        )

        allowed_periods = {"1y", "2y", "5y"}
        if period not in allowed_periods:
            logger.warning(
                "get_historical_prices: unsupported period=%s; defaulting to 2y",
                period,
            )
            period = "2y"

        benchmark = BENCHMARK_TICKER or "SPY"
        all_symbols = [str(t).upper().strip() for t in (tickers or []) if t]
        if benchmark:
            all_symbols_with_benchmark = all_symbols + [benchmark]
        else:
            all_symbols_with_benchmark = all_symbols

        # De-duplicate while preserving order
        seen = set()
        ordered_symbols: List[str] = []
        for s in all_symbols_with_benchmark:
            if s not in seen:
                ordered_symbols.append(s)
                seen.add(s)

        prices: Dict[str, DataFrame] = {}
        for symbol in ordered_symbols:
            try:
                logger.info("get_historical_prices: downloading %s", symbol)

                try:
                    import yfinance as yf
                except ImportError:
                    logger.exception("get_historical_prices: yfinance not available")
                    prices[symbol] = pd.DataFrame(
                        columns=["date", "open", "high", "low", "close", "volume"]
                    )
                    continue

                df_raw = yf.download(
                    symbol,
                    period=period,
                    progress=False,
                    auto_adjust=False,
                    actions=False,
                    group_by="column",
                    interval="1d",
                )
                df = self._normalize_yfinance_ohlcv(df_raw)

                # Persist even if empty; DB method will no-op.
                try:
                    self.db.upsert_prices(df, symbol)
                except Exception as e:
                    logger.exception(
                        "get_historical_prices: failed upserting %s prices: %s",
                        symbol,
                        str(e),
                    )

                prices[symbol] = df
            except Exception as e:
                logger.warning(
                    "get_historical_prices: missing/invalid data for %s: %s",
                    symbol,
                    str(e),
                )
                prices[symbol] = pd.DataFrame(
                    columns=["date", "open", "high", "low", "close", "volume"]
                )

        logger.info("get_historical_prices: finished")
        return prices

    def get_news(self, tickers: List[str]) -> Dict[str, List[str]]:
        """Fetch recent news headlines for each ticker and persist them."""
        logger.info("get_news: starting (tickers=%s)", tickers)
        if not self._robinhood_logged_in:
            logger.warning("get_news: Robinhood not logged in; returning empty news")
            return {}

        if not tickers:
            logger.info("get_news: no tickers provided")
            return {}

        try:
            from robin_stocks.robinhood.stocks import get_news as rh_get_news
        except ImportError as e:
            logger.exception("get_news: robin_stocks not available: %s", str(e))
            return {}

        results: Dict[str, List[str]] = {}

        for ticker in tickers:
            symbol = str(ticker).upper().strip()
            if not symbol:
                continue

            try:
                logger.info("get_news: fetching news for %s", symbol)
                items = rh_get_news(symbol) or []

                parsed: List[Dict[str, Any]] = []
                for n in items:
                    if not isinstance(n, dict):
                        continue
                    headline = n.get("title")
                    source = n.get("source")
                    published_raw = n.get("published_at")
                    published_at = pd.to_datetime(published_raw, errors="coerce")
                    published_at_py = (
                        published_at.to_pydatetime()
                        if published_at is not pd.NaT and published_at is not None
                        else None
                    )
                    if headline is None:
                        continue
                    parsed.append(
                        {
                            "headline": str(headline),
                            "source": str(source) if source else None,
                            "published_at": published_at_py,
                        }
                    )

                # Sort by published_at desc (None last) and take 10.
                parsed.sort(
                    key=lambda x: x["published_at"] or pd.Timestamp(0), reverse=True
                )
                parsed = parsed[:10]

                try:
                    self.db.save_news(symbol, parsed)
                except Exception as e:
                    logger.exception(
                        "get_news: failed saving news for %s: %s", symbol, str(e)
                    )

                results[symbol] = [p["headline"] for p in parsed]
            except Exception as e:
                logger.warning("get_news: failed for %s: %s", symbol, str(e))
                results[symbol] = []

        logger.info("get_news: finished")
        return results

    def get_all_data(self) -> Dict[str, Any]:
        """
        Main entry point for other phases.

        Returns:
            {
              "portfolio": [...],
              "prices": {ticker: df},
              "news": {ticker: [headlines]},
              "benchmark": spy_dataframe
            }
        """
        logger.info("get_all_data: starting")

        import os
        from datetime import datetime

        imported_path = "outputs/imported_portfolio.json"

        if os.path.exists(imported_path):
            try:
                with open(imported_path) as f:
                    imported = json.load(f)

                imported_at_str = imported.get("imported_at", "")
                imported_at = (
                    datetime.fromisoformat(imported_at_str)
                    if imported_at_str
                    else None
                )

                # Use imported if it exists and is < 24h old
                if imported_at:
                    age_hours = (
                        (datetime.now() - imported_at).total_seconds() / 3600
                    )
                    if age_hours < 24:
                        positions = imported.get("positions", [])
                        if positions:
                            source = imported.get("source", "imported")
                            self.logger.info(
                                "get_all_data: using imported portfolio (%s, %.1fh old, %d positions)",
                                source,
                                age_hours,
                                len(positions),
                            )
                            # Fetch prices for imported tickers
                            tickers = list({
                                p["ticker"]
                                for p in positions
                                if p.get("ticker")
                            })
                            prices = self.get_historical_prices(
                                tickers,
                                period="2y",
                            )
                            news = {}
                            for t in tickers:
                                try:
                                    news[t] = (
                                        self.get_news([t]).get(t, [])
                                    )
                                except Exception:
                                    news[t] = []

                            return {
                                "portfolio": positions,
                                "prices": prices,
                                "news": news,
                                "benchmark": prices.get(
                                    "SPY",
                                    self.get_historical_prices(
                                        ["SPY"], "5y"
                                    ).get("SPY", pd.DataFrame()),
                                ),
                            }
                    else:
                        self.logger.info(
                            "get_all_data: imported portfolio is %.1fh old — falling back to Robinhood",
                            age_hours,
                        )
            except Exception as e:
                self.logger.warning(
                    "get_all_data: could not read imported portfolio: %s", e
                )

        portfolio: List[Dict[str, Any]] = []
        news: Dict[str, List[str]] = {}
        prices_all: Dict[str, DataFrame] = {}

        benchmark = BENCHMARK_TICKER or "SPY"

        robin_ok = False
        try:
            robin_ok = self.login_robinhood()
        except Exception as e:
            logger.exception("get_all_data: login_robinhood raised: %s", str(e))
            robin_ok = False

        try:
            if robin_ok:
                portfolio = self.get_portfolio()
                tickers = [p.get("ticker") for p in portfolio if p.get("ticker")]
                news = self.get_news(tickers)
                prices_all = self.get_historical_prices(tickers, period="2y")
            else:
                logger.warning(
                    "get_all_data: Robinhood login failed; fetching prices only via yfinance"
                )

                # Portfolio/news can't be fetched from Robinhood. Use latest cached portfolio
                # snapshot (DB) only to determine tickers for price downloads.
                tickers_from_cache: List[str] = []
                try:
                    latest_df = self.db.get_latest_portfolio()
                    if latest_df is not None and not latest_df.empty:
                        portfolio = latest_df.to_dict("records")
                        tickers_from_cache = [
                            str(x).upper().strip()
                            for x in latest_df.get("ticker", []).tolist()
                            if x
                        ]
                except Exception as e:
                    logger.exception(
                        "get_all_data: failed reading cached portfolio tickers: %s",
                        str(e),
                    )

                prices_all = self.get_historical_prices(
                    tickers_from_cache, period="2y"
                )
        except Exception as e:
            logger.exception("get_all_data: exception while assembling data: %s", str(e))

        spy_df = prices_all.get(benchmark, pd.DataFrame())
        prices = {k: v for k, v in prices_all.items() if k != benchmark}

        output = {
            "portfolio": portfolio,
            "prices": prices,
            "news": news,
            "benchmark": spy_df,
        }
        logger.info("get_all_data: finished")
        return output
