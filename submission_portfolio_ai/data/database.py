from __future__ import annotations

import logging
from datetime import date as DateType
from datetime import datetime as DateTimeType
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    select,
    text,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import declarative_base, sessionmaker

from config.settings import DB_PATH

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

Base = declarative_base()


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False)
    quantity = Column(Float)
    average_buy_price = Column(Float)
    current_price = Column(Float)
    market_value = Column(Float)
    unrealized_pnl = Column(Float)
    unrealized_pnl_pct = Column(Float)
    snapshot_date = Column(DateTime, server_default=func.now(), nullable=False)
    hrp_optimal_weight = Column(Float, nullable=True)
    current_weight = Column(Float, nullable=True)


class HistoricalPrice(Base):
    __tablename__ = "historical_prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(BigInteger)

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_historical_prices_ticker_date"),
    )


class NewsItem(Base):
    __tablename__ = "news_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False)
    headline = Column(Text)
    source = Column(String)
    published_at = Column(DateTime)
    sentiment_label = Column(String, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    fetched_at = Column(DateTime, server_default=func.now(), nullable=False)


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_date = Column(DateTime, server_default=func.now(), nullable=False)
    tickers_analyzed = Column(String)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    var_95 = Column(Float)
    committee_decision = Column(Text)
    status = Column(String)
    regime_label = Column(String, nullable=True)
    regime_probabilities = Column(String, nullable=True)  # JSON string
    risk_scalar = Column(Float, nullable=True)
    bull_agent_output = Column(Text, nullable=True)
    bear_agent_output = Column(Text, nullable=True)
    cro_agent_output = Column(Text, nullable=True)
    portfolio_risk_score = Column(Integer, nullable=True)
    executive_summary = Column(Text, nullable=True)
    pdf_path = Column(String, nullable=True)
    var_95_proposed = Column(Float, nullable=True)
    max_drawdown_proposed = Column(Float, nullable=True)
    sharpe_proposed = Column(Float, nullable=True)
    beta_proposed = Column(Float, nullable=True)
    monte_carlo_5th_pct_proposed = Column(Float, nullable=True)
    post_rebalance_flags = Column(Text, nullable=True)
    backtest_hrp_cagr = Column(Float, nullable=True)
    backtest_hrp_sharpe = Column(Float, nullable=True)
    backtest_benchmark_cagr = Column(Float, nullable=True)


class TickerMetrics(Base):
    __tablename__ = "ticker_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("analysis_runs.id"), nullable=True)
    ticker = Column(String, nullable=False)
    garch_annualized_vol = Column(Float, nullable=True)
    garch_vol_regime = Column(String, nullable=True)
    atr_14 = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    hrp_weight = Column(Float, nullable=True)
    current_weight = Column(Float, nullable=True)
    weight_delta = Column(Float, nullable=True)
    rebalance_action = Column(String, nullable=True)
    analyst_target_price = Column(Float, nullable=True)
    analyst_upside_pct = Column(Float, nullable=True)
    analyst_signal = Column(String, nullable=True)
    marginal_cvar_contribution = Column(Float, nullable=True)
    concentration_risk_flag = Column(Boolean, nullable=True)
    trend_direction = Column(String, nullable=True)
    trend_slope_normalized = Column(Float, nullable=True)
    trend_slope_acceleration = Column(Float, nullable=True)
    trend_deviation_std = Column(Float, nullable=True)
    trend_uncertainty_pct = Column(Float, nullable=True)
    seasonal_component_pct = Column(Float, nullable=True)
    trend_confidence_score = Column(Float, nullable=True)
    computed_at = Column(DateTime, default=DateTimeType.utcnow)


class RegimeHistory(Base):
    __tablename__ = "regime_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True)
    regime_label = Column(String, nullable=True)
    bull_probability = Column(Float, nullable=True)
    bear_probability = Column(Float, nullable=True)
    crash_probability = Column(Float, nullable=True)
    risk_scalar = Column(Float, nullable=True)
    spy_return = Column(Float, nullable=True)
    vix_close = Column(Float, nullable=True)


class DatabaseManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        self.Session = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
        )

    @staticmethod
    def _coerce_to_date(value: Any) -> DateType:
        if isinstance(value, DateType) and not isinstance(value, DateTimeType):
            return value
        if isinstance(value, DateTimeType):
            return value.date()
        return pd.to_datetime(value).date()

    def init_db(self) -> None:
        """Create all tables if they do not exist."""
        try:
            Base.metadata.create_all(self.engine)

            # Migrate existing tables — add new columns if missing
            with self.engine.connect() as conn:
                existing = {
                    row[1]
                    for row in conn.execute(
                        text("PRAGMA table_info(portfolio_snapshots)")
                    )
                }
                if "hrp_optimal_weight" not in existing:
                    conn.execute(
                        text(
                            "ALTER TABLE portfolio_snapshots "
                            "ADD COLUMN hrp_optimal_weight REAL"
                        )
                    )
                if "current_weight" not in existing:
                    conn.execute(
                        text(
                            "ALTER TABLE portfolio_snapshots "
                            "ADD COLUMN current_weight REAL"
                        )
                    )

                existing2 = {
                    row[1]
                    for row in conn.execute(text("PRAGMA table_info(analysis_runs)"))
                }
                new_analysis_cols = {
                    "regime_label": "TEXT",
                    "regime_probabilities": "TEXT",
                    "risk_scalar": "REAL",
                    "bull_agent_output": "TEXT",
                    "bear_agent_output": "TEXT",
                    "cro_agent_output": "TEXT",
                    "portfolio_risk_score": "INTEGER",
                    "executive_summary": "TEXT",
                    "pdf_path": "TEXT",
                    "var_95_proposed": "REAL",
                    "max_drawdown_proposed": "REAL",
                    "sharpe_proposed": "REAL",
                    "beta_proposed": "REAL",
                    "monte_carlo_5th_pct_proposed": "REAL",
                    "post_rebalance_flags": "TEXT",
                    "backtest_hrp_cagr": "REAL",
                    "backtest_hrp_sharpe": "REAL",
                    "backtest_benchmark_cagr": "REAL",
                }
                for col, col_type in new_analysis_cols.items():
                    if col not in existing2:
                        conn.execute(
                            text(
                                f"ALTER TABLE analysis_runs "
                                f"ADD COLUMN {col} {col_type}"
                            )
                        )

                existing3 = {
                    row[1]
                    for row in conn.execute(text("PRAGMA table_info(ticker_metrics)"))
                }
                new_ticker_metric_cols = {
                    "trend_direction": "TEXT",
                    "trend_slope_normalized": "REAL",
                    "trend_slope_acceleration": "REAL",
                    "trend_deviation_std": "REAL",
                    "trend_uncertainty_pct": "REAL",
                    "seasonal_component_pct": "REAL",
                    "trend_confidence_score": "REAL",
                }
                for col, col_type in new_ticker_metric_cols.items():
                    if col not in existing3:
                        conn.execute(
                            text(
                                f"ALTER TABLE ticker_metrics "
                                f"ADD COLUMN {col} {col_type}"
                            )
                        )
                conn.commit()

            logger.info("Initialized database tables in %s", self.db_path)
        except Exception:
            logger.exception("init_db failed for %s", self.db_path)
            raise

    def upsert_prices(self, df: pd.DataFrame, ticker: str) -> int:
        """Insert or update prices in `historical_prices` for (ticker, date)."""
        if df is None or df.empty:
            logger.info("upsert_prices: empty dataframe for ticker=%s", ticker)
            return 0

        required_cols = {"date", "open", "high", "low", "close", "volume"}
        missing = sorted(required_cols - set(df.columns))
        if missing:
            raise ValueError(f"upsert_prices missing columns for {ticker}: {missing}")

        try:
            tmp = df.copy()
            tmp["ticker"] = ticker
            tmp["date"] = pd.to_datetime(tmp["date"]).dt.date
            tmp["open"] = pd.to_numeric(tmp["open"], errors="coerce")
            tmp["high"] = pd.to_numeric(tmp["high"], errors="coerce")
            tmp["low"] = pd.to_numeric(tmp["low"], errors="coerce")
            tmp["close"] = pd.to_numeric(tmp["close"], errors="coerce")
            tmp["volume"] = (
                pd.to_numeric(tmp["volume"], errors="coerce").fillna(0).astype("int64")
            )

            records = tmp[
                ["ticker", "date", "open", "high", "low", "close", "volume"]
            ].to_dict("records")

            if not records:
                return 0

            stmt = sqlite_insert(HistoricalPrice).values(records)
            update_cols = {
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "volume": stmt.excluded.volume,
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=["ticker", "date"],
                set_=update_cols,
            )

            with self.Session() as session:
                logger.info(
                    "upsert_prices: upserting %d rows for ticker=%s",
                    len(records),
                    ticker,
                )
                result = session.execute(stmt)
                session.commit()

                rowcount = result.rowcount
                return int(len(records) if rowcount in (None, -1) else rowcount)
        except Exception:
            logger.exception("upsert_prices failed for ticker=%s", ticker)
            raise

    def save_portfolio_snapshot(self, positions_list: Sequence[Dict[str, Any]]) -> None:
        """Save portfolio snapshot positions to `portfolio_snapshots`."""
        if not positions_list:
            logger.info("save_portfolio_snapshot: no positions to save")
            return

        try:
            snapshots: List[PortfolioSnapshot] = []
            for pos in positions_list:
                snapshots.append(
                    PortfolioSnapshot(
                        ticker=pos["ticker"],
                        quantity=pos.get("quantity"),
                        average_buy_price=pos.get("average_buy_price"),
                        current_price=pos.get("current_price"),
                        market_value=pos.get("market_value"),
                        unrealized_pnl=pos.get("unrealized_pnl"),
                        unrealized_pnl_pct=pos.get("unrealized_pnl_pct"),
                    )
                )

            with self.Session() as session:
                logger.info(
                    "save_portfolio_snapshot: inserting %d positions",
                    len(snapshots),
                )
                session.add_all(snapshots)
                session.flush()
                session.commit()
        except Exception:
            logger.exception("save_portfolio_snapshot failed")
            raise

    def save_news(self, ticker: str, news_list: Iterable[Dict[str, Any]]) -> None:
        """Save fetched/enriched news items to `news_items`."""
        try:
            items = list(news_list) if news_list is not None else []
            if not items:
                logger.info("save_news: no news to save for ticker=%s", ticker)
                return

            rows: List[NewsItem] = []
            for n in items:
                rows.append(
                    NewsItem(
                        ticker=ticker,
                        headline=n.get("headline"),
                        source=n.get("source"),
                        published_at=n.get("published_at"),
                        sentiment_label=n.get("sentiment_label"),
                        sentiment_score=n.get("sentiment_score"),
                    )
                )

            with self.Session() as session:
                logger.info("save_news: inserting %d items for ticker=%s", len(rows), ticker)
                session.add_all(rows)
                session.commit()
        except Exception:
            logger.exception("save_news failed for ticker=%s", ticker)
            raise

    def get_prices(
        self,
        ticker: str,
        start_date: Any,
        end_date: Any,
    ) -> pd.DataFrame:
        """Fetch historical prices between `start_date` and `end_date` inclusive."""
        try:
            start_dt = self._coerce_to_date(start_date)
            end_dt = self._coerce_to_date(end_date)

            with self.Session() as session:
                query = (
                    select(HistoricalPrice)
                    .where(HistoricalPrice.ticker == ticker)
                    .where(HistoricalPrice.date >= start_dt)
                    .where(HistoricalPrice.date <= end_dt)
                    .order_by(HistoricalPrice.date.asc())
                )
                rows = session.execute(query).scalars().all()

            if not rows:
                return pd.DataFrame(
                    columns=["date", "open", "high", "low", "close", "volume"]
                )

            data = [
                {
                    "ticker": r.ticker,
                    "date": r.date,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                }
                for r in rows
            ]
            return pd.DataFrame(data)
        except Exception:
            logger.exception(
                "get_prices failed for ticker=%s start=%s end=%s",
                ticker,
                start_date,
                end_date,
            )
            raise

    def get_latest_portfolio(self) -> pd.DataFrame:
        """Fetch the most recent portfolio snapshot rows from `portfolio_snapshots`."""
        portfolio_columns = [
            "id",
            "ticker",
            "quantity",
            "average_buy_price",
            "current_price",
            "market_value",
            "unrealized_pnl",
            "unrealized_pnl_pct",
            "snapshot_date",
        ]
        empty_df = pd.DataFrame(columns=portfolio_columns)
        try:
            # Compare via SQL subquery so SQLite matches stored datetimes (Python bind
            # values can include microseconds that do not match TEXT rows in SQLite).
            max_snapshot_subq = select(func.max(PortfolioSnapshot.snapshot_date)).scalar_subquery()
            query = (
                select(PortfolioSnapshot)
                .where(PortfolioSnapshot.snapshot_date == max_snapshot_subq)
                .order_by(PortfolioSnapshot.ticker.asc())
            )
            with self.Session() as session:
                rows = session.execute(query).scalars().all()

            if not rows:
                return empty_df.copy()

            data = [
                {
                    "id": r.id,
                    "ticker": r.ticker,
                    "quantity": r.quantity,
                    "average_buy_price": r.average_buy_price,
                    "current_price": r.current_price,
                    "market_value": r.market_value,
                    "unrealized_pnl": r.unrealized_pnl,
                    "unrealized_pnl_pct": r.unrealized_pnl_pct,
                    "snapshot_date": r.snapshot_date,
                }
                for r in rows
            ]
            return pd.DataFrame(data, columns=portfolio_columns)
        except Exception:
            logger.exception("get_latest_portfolio failed")
            raise

    def save_analysis_run(self, metrics_dict: Dict[str, Any]) -> None:
        """Save a single analysis run to `analysis_runs`."""
        try:
            if not metrics_dict:
                raise ValueError("save_analysis_run requires a non-empty metrics_dict")

            def _coerce_float(value: Any) -> Any:
                if value is None:
                    return None
                if isinstance(value, str):
                    cleaned = value.strip().replace("%", "")
                    if cleaned == "":
                        return None
                    try:
                        parsed = float(cleaned)
                        # Percent-formatted drawdowns should be stored as decimals.
                        if "%" in value:
                            return parsed / 100.0
                        return parsed
                    except ValueError:
                        return value
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return value

            with self.Session() as session:
                run = AnalysisRun(
                    tickers_analyzed=metrics_dict.get("tickers_analyzed"),
                    sharpe_ratio=metrics_dict.get("sharpe_ratio"),
                    max_drawdown=_coerce_float(metrics_dict.get("max_drawdown")),
                    var_95=metrics_dict.get("var_95"),
                    committee_decision=metrics_dict.get("committee_decision"),
                    status=metrics_dict.get("status"),
                )
                logger.info("save_analysis_run: inserting analysis run")
                session.add(run)
                session.commit()
        except Exception:
            logger.exception("save_analysis_run failed")
            raise

    def update_portfolio_hrp_weights(
        self,
        hrp_weights: dict,
        current_weights: dict,
    ) -> None:
        """
        Update hrp_optimal_weight and current_weight columns
        in the most recent portfolio_snapshots rows.

        hrp_weights:     {ticker: float} from HRPEngine
        current_weights: {ticker: float} from HRPEngine
        """
        with self.Session() as session:
            for ticker, hrp_w in (hrp_weights or {}).items():
                curr_w = (current_weights or {}).get(ticker)

                # Find most recent snapshot for this ticker
                stmt = (
                    select(PortfolioSnapshot)
                    .where(PortfolioSnapshot.ticker == ticker)
                    .order_by(PortfolioSnapshot.snapshot_date.desc())
                    .limit(1)
                )
                row = session.execute(stmt).scalar_one_or_none()

                if row is not None:
                    row.hrp_optimal_weight = round(float(hrp_w), 6)
                    row.current_weight = (
                        round(float(curr_w), 6) if curr_w is not None else None
                    )
                    logger.info(
                        "update_portfolio_hrp_weights: %s hrp=%.4f current=%.4f",
                        ticker,
                        hrp_w,
                        curr_w or 0,
                    )
                else:
                    logger.warning(
                        "update_portfolio_hrp_weights: no snapshot found for %s",
                        ticker,
                    )
            session.commit()

    def save_ticker_metrics(self, metrics: Dict[str, Any]) -> int:
        """Save one ticker_metrics row and return inserted id."""
        if not metrics:
            raise ValueError("save_ticker_metrics requires a non-empty metrics dict")

        with self.Session() as session:
            row = TickerMetrics(**metrics)
            session.add(row)
            session.flush()
            row_id = int(row.id)
            session.commit()
            logger.info("save_ticker_metrics: %s", metrics.get("ticker"))
            return row_id

    def save_regime_history(self, record: Dict[str, Any]) -> None:
        """
        Save one regime_history row using INSERT OR IGNORE on unique date.
        """
        if not record:
            raise ValueError("save_regime_history requires a non-empty record dict")

        payload = dict(record)
        if "date" in payload and payload["date"] is not None:
            payload["date"] = self._coerce_to_date(payload["date"])

        stmt = sqlite_insert(RegimeHistory).values(payload).prefix_with("OR IGNORE")

        with self.Session() as session:
            session.execute(stmt)
            session.commit()
            logger.info(
                "save_regime_history: %s %s",
                payload.get("date"),
                payload.get("regime_label"),
            )
