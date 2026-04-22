import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime
from typing import Optional

from sqlalchemy import select

from ai.committee import InvestmentCommittee
from ai.committee_inputs import CommitteeInputs
from config.settings import DB_PATH, DEFAULT_LLM, TRADING_DAYS
from data.analyst_fetcher import AnalystFetcher
from data.data_loader import DataLoader
from data.database import AnalysisRun, DatabaseManager
from data.validator import DataValidator
from quant import QuantEngine
from quant.garch_engine import GARCHEngine
from quant.hrp_engine import HRPEngine
from quant.portfolio import PortfolioOptimizer
from quant.post_rebalance_engine import PostRebalanceEngine, validate_cro_claims
from quant.regime_engine import RegimeEngine
from quant.risk_levels import RiskLevelCalculator
from quant.trend_engine import TrendEngine

os.makedirs("outputs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/pipeline.log", mode="a"),
    ],
)
logger = logging.getLogger("main")


def _parse_pct(val) -> Optional[float]:
    """Convert '-21.3%' -> -0.213 or return None."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val.replace("%", "")) / 100
        except ValueError:
            return None
    return None


def step_load_data(args, db) -> dict:
    """Phase 1: Load portfolio + prices from Robinhood/yfinance into SQLite."""
    t0 = time.time()
    logger.info("=" * 50)
    logger.info("STEP 1: DATA INGESTION")
    logger.info("=" * 50)

    try:
        if args.no_refresh:
            logger.info("--no-refresh: reading from DB only")
            portfolio_df = db.get_latest_portfolio()
            portfolio = portfolio_df.to_dict("records")
            all_tickers = list({p.get("ticker") for p in portfolio if p.get("ticker")})
            prices = {}
            for ticker in all_tickers:
                df = db.get_prices(ticker, "2020-01-01", "2030-12-31")
                if not df.empty:
                    prices[ticker] = df
            spy = db.get_prices("SPY", "2020-01-01", "2030-12-31")
        else:
            loader = DataLoader(db_manager=db)
            raw = loader.get_all_data()

            # Optional period override
            if args.price_period != "2y":
                tickers = [p.get("ticker") for p in (raw.get("portfolio") or []) if p.get("ticker")]
                refreshed = loader.get_historical_prices(tickers, period=args.price_period)
                raw["benchmark"] = refreshed.get("SPY")
                raw["prices"] = {k: v for k, v in refreshed.items() if k != "SPY"}

            validator = DataValidator()
            clean = validator.validate_all(raw)
            portfolio = clean["portfolio"]
            prices = clean["prices"]
            spy = clean["benchmark"]

            # Ensure SPY has 5 years for HMM
            if len(spy) < 756:
                logger.warning("SPY history short (%d rows), fetching 5y", len(spy))
                import pandas as pd
                import yfinance as yf

                df = yf.download(
                    "SPY",
                    period="5y",
                    interval="1d",
                    progress=False,
                    auto_adjust=False,
                )
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                df = df.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )
                df = df.reset_index().rename(columns={"Date": "date"})
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
                df = df.dropna(subset=["date", "close"])
                db.upsert_prices(df, "SPY")
                spy = db.get_prices("SPY", "2020-01-01", "2030-12-31")

        logger.info(
            "Data loaded: %d positions, %d tickers, SPY=%d rows (%.1fs)",
            len(portfolio),
            len(prices),
            len(spy),
            time.time() - t0,
        )
        return {"portfolio": portfolio, "prices": prices, "spy": spy}
    except Exception as e:
        logger.error("STEP 1 failed: %s", e, exc_info=True)
        raise


def step_quant(data_out, db) -> dict:
    """Phase 2: Run full quant engine."""
    t0 = time.time()
    logger.info("=" * 50)
    logger.info("STEP 2: QUANT ENGINE")
    logger.info("=" * 50)

    try:
        prices = data_out["prices"]
        portfolio = data_out["portfolio"]
        spy = data_out["spy"]

        engine = QuantEngine(
            {
                "prices": prices,
                "portfolio": portfolio,
                "benchmark": spy,
                "news": {},
            }
        )
        engine.run()
        quant_ci = engine.get_committee_input()

        logger.info(
            "Quant complete: sharpe=%.3f beta=%.3f drawdown=%s (%.1fs)",
            quant_ci.get("portfolio_sharpe", 0) or 0,
            quant_ci.get("portfolio_beta", 0) or 0,
            quant_ci.get("portfolio_max_drawdown_pct", "N/A"),
            time.time() - t0,
        )
        return {"quant_ci": quant_ci}
    except Exception as e:
        logger.error("STEP 2 failed: %s", e, exc_info=True)
        raise


def step_advanced_quant(data_out, args=None) -> dict:
    """Phase 4: GARCH + HMM + HRP + Trend."""
    t0 = time.time()
    logger.info("=" * 50)
    logger.info("STEP 4: ADVANCED QUANT MODELS")
    logger.info("=" * 50)

    try:
        prices = data_out["prices"]
        portfolio = data_out["portfolio"]
        spy = data_out["spy"]

        # 4A: GARCH
        logger.info("  4A: GARCH volatility forecasting")
        garch_engine = GARCHEngine(prices)
        garch_output = garch_engine.run_all()
        garch_ci = garch_engine.get_committee_input()

        # 4B: HMM Regime
        logger.info("  4B: HMM regime detection")
        regime_engine = RegimeEngine(spy)
        regime_output = regime_engine.run()
        regime_current = regime_output["current"]
        logger.info(
            "  Regime: %s (scalar=%.3f)",
            regime_current.get("dominant_regime"),
            regime_current.get("risk_scalar", 0),
        )

        # 4C: HRP
        logger.info("  4C: HRP portfolio optimization")
        mc_output = PortfolioOptimizer(prices, portfolio).monte_carlo()
        hrp_engine = HRPEngine(prices, portfolio, monte_carlo_output=mc_output)
        hrp_output = hrp_engine.run()

        concentration_risk_ticker = None
        for ticker, flagged in (
            (hrp_output.get("cvar_analysis") or {}).get("concentration_flags", {}).items()
        ):
            if flagged:
                concentration_risk_ticker = ticker
                break
        hrp_ci = {
            "rebalancing_table": hrp_output.get("rebalancing_table"),
            "concentration_risk_ticker": concentration_risk_ticker,
            "portfolio_cvar": (hrp_output.get("cvar_analysis") or {}).get("portfolio_cvar"),
            "hrp_weights": hrp_output.get("hrp_weights"),
        }

        # 4D: Trend Engine
        logger.info("  4D: Trend linear regression signals")
        trend_engine = TrendEngine(prices)
        trend_output = trend_engine.run_all()
        trend_ci = trend_engine.get_committee_input()
        low_conf = trend_engine.get_low_confidence_tickers()

        # Backtest engine
        backtest_ci = {}
        if not getattr(args, "skip_backtest", False):
            logger.info("  Backtest: walk-forward engine")
            try:
                from quant.backtest_engine import BacktestEngine
                import time as _time

                t0_bt = _time.time()
                backtest_prices = prices
                lengths = [len(df) for df in prices.values() if df is not None]
                shortest = min(lengths) if lengths else 0

                # Backtest needs deeper history than 2y to produce enough walk-forward periods.
                # Fetch 5y ticker prices for backtest only when current series are short.
                if shortest < 756:
                    logger.info(
                        "  Backtest: ticker history short (min=%d rows), fetching 5y prices for backtest",
                        shortest,
                    )
                    bt_loader = DataLoader(db_manager=DatabaseManager())
                    bt_all = bt_loader.get_historical_prices(
                        list(prices.keys()),
                        period="5y",
                    )
                    backtest_prices = {k: v for k, v in bt_all.items() if k != "SPY"}

                bt = BacktestEngine(
                    prices=backtest_prices,
                    portfolio=portfolio,
                    spy=spy,
                    train_days=252,
                    test_days=63,
                    regime_scalar_cap=0.5,
                )
                backtest_ci = bt.run()
                logger.info(
                    "  Backtest complete: %d periods HRP_CAGR=%.2f%% BM_CAGR=%.2f%%  (%.1fs)",
                    backtest_ci.get("n_periods", 0),
                    backtest_ci.get("hrp_strategy_cagr", 0) * 100,
                    backtest_ci.get("benchmark_cagr", 0) * 100,
                    _time.time() - t0_bt,
                )
                logger.info(
                    "  Backtest validity: %s",
                    backtest_ci.get("validity_note", "N/A"),
                )
                logger.info(
                    "  Drawdown reduction: %.1f%%",
                    backtest_ci.get("drawdown_reduction_pct", 0) * 100,
                )
            except Exception as e:
                logger.error("  Backtest failed: %s", e)
        else:
            logger.info("--skip-backtest: skipping")

        logger.info("Advanced quant complete (%.1fs)", time.time() - t0)
        return {
            "garch_output": garch_output,
            "garch_ci": garch_ci,
            "regime_current": regime_current,
            "hrp_output": hrp_output,
            "hrp_ci": hrp_ci,
            "mc_output": mc_output,
            "trend_output": trend_output,
            "trend_ci": trend_ci,
            "low_conf": low_conf,
            "backtest_ci": backtest_ci,
        }
    except Exception as e:
        logger.error("STEP 4 failed: %s", e, exc_info=True)
        raise


def step_external_data(data_out) -> dict:
    """Phase 5: Analyst targets + ATR stops."""
    t0 = time.time()
    logger.info("=" * 50)
    logger.info("STEP 5: EXTERNAL DATA ENRICHMENT")
    logger.info("=" * 50)

    try:
        portfolio = data_out["portfolio"]
        prices = data_out["prices"]
        garch_output = data_out["garch_output"]

        # 5A: Analyst targets
        fetcher = AnalystFetcher(portfolio)
        analyst_output = fetcher.run_all()

        # 5B: ATR stop levels
        calc = RiskLevelCalculator(
            prices=prices,
            portfolio=portfolio,
            garch_output=garch_output,
        )
        risk_output = calc.run_all()

        logger.info("External data complete (%.1fs)", time.time() - t0)
        return {"analyst_output": analyst_output, "risk_output": risk_output}
    except Exception as e:
        logger.error("STEP 5 failed: %s", e, exc_info=True)
        raise


def step_committee(data_out, quant_out, adv_out, ext_out, args) -> dict:
    """Phase 6: Multi-agent committee."""
    t0 = time.time()
    logger.info("=" * 50)
    logger.info("STEP 6: AI INVESTMENT COMMITTEE")
    logger.info("=" * 50)

    try:
        portfolio = data_out["portfolio"]
        prices = data_out["prices"]
        spy = data_out["spy"]

        inputs_obj = CommitteeInputs(
            quant_committee_input=quant_out["quant_ci"],
            hrp_committee_input=adv_out["hrp_ci"],
            regime_output=adv_out["regime_current"],
            garch_committee_input=adv_out["garch_ci"],
            stop_levels=ext_out["risk_output"],
            analyst_targets=ext_out["analyst_output"],
            portfolio=portfolio,
        )
        inputs = inputs_obj.build()
        inputs["trend_signals"] = adv_out["trend_ci"]
        inputs["low_confidence_tickers"] = adv_out["low_conf"]
        backtest_ci = adv_out.get("backtest_ci", {})
        if backtest_ci:
            inputs["backtest_results"] = {
                "hrp_cagr": backtest_ci.get("hrp_strategy_cagr"),
                "benchmark_cagr": backtest_ci.get("benchmark_cagr"),
                "hrp_sharpe": backtest_ci.get("hrp_sharpe"),
                "benchmark_sharpe": backtest_ci.get("benchmark_sharpe"),
                "hrp_max_drawdown": backtest_ci.get("hrp_max_drawdown"),
                "benchmark_max_drawdown": backtest_ci.get("benchmark_max_drawdown"),
                "regime_timing_value": backtest_ci.get("regime_timing_value"),
                "n_periods": backtest_ci.get("n_periods"),
            }

        if args.skip_committee:
            logger.warning("--skip-committee: returning empty verdict")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_path = os.path.join(args.output_dir, "latest_run.json")
            cache_payload = {
                "timestamp": timestamp,
                "inputs": inputs,
                "verdict": {},
                "proposed_weights": {},
                "post_rebalance": None,
                "validation_flags": [],
                "model_used": args.model,
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_payload, f, indent=2, default=str)
            return {
                "verdict": {},
                "proposed_weights": {},
                "post_rebalance": None,
                "validation_flags": [],
                "cache_path": cache_path,
                "committee": None,
                "inputs": inputs,
            }

        committee = InvestmentCommittee(
            model_key=args.model,
            prices=prices,
            benchmark=spy,
            portfolio=portfolio,
        )
        verdict = committee.run(inputs)
        cache_path = committee.save_cache(inputs, output_dir=args.output_dir)

        logger.info(
            "Committee complete: risk_score=%s flags=%d (%.1fs)",
            verdict.get("portfolio_risk_score", "N/A"),
            len(committee.validation_flags),
            time.time() - t0,
        )
        return {
            "verdict": verdict,
            "proposed_weights": committee.proposed_weights,
            "post_rebalance": committee.post_rebalance,
            "validation_flags": committee.validation_flags,
            "cache_path": cache_path,
            "committee": committee,
            "inputs": inputs,
        }
    except Exception as e:
        logger.error("STEP 6 failed: %s", e, exc_info=True)
        raise


def step_save_to_db(data_out, quant_out, adv_out, ext_out, committee_out, db) -> None:
    """Save all results to SQLite tables."""
    t0 = time.time()
    logger.info("=" * 50)
    logger.info("STEP DB: SAVING TO SQLITE")
    logger.info("=" * 50)

    try:
        prices = data_out["prices"]
        garch_output = adv_out["garch_output"]
        hrp_output = adv_out["hrp_output"]
        regime_curr = adv_out["regime_current"]
        trend_output = adv_out.get("trend_output", {})
        risk_output = ext_out["risk_output"]
        analyst_out = ext_out["analyst_output"]
        post_reb = committee_out.get("post_rebalance")
        flags = committee_out.get("validation_flags", [])
        quant_ci = quant_out["quant_ci"]

        # Save HRP weights to portfolio_snapshots
        db.update_portfolio_hrp_weights(
            hrp_weights=hrp_output.get("hrp_weights", {}),
            current_weights=hrp_output.get("current_weights", {}),
        )
        logger.info("  HRP weights saved to portfolio_snapshots")

        # Build analysis run fields
        committee_obj = committee_out.get("committee")
        db_fields = (
            committee_obj.get_db_fields({"current": regime_curr})
            if committee_obj and hasattr(committee_obj, "get_db_fields")
            else {}
        )

        if post_reb:
            after = post_reb.get("after", {})
            db_fields.update(
                {
                    "var_95_proposed": after.get("var_95"),
                    "max_drawdown_proposed": after.get("max_drawdown"),
                    "sharpe_proposed": after.get("sharpe"),
                    "beta_proposed": after.get("beta"),
                    "monte_carlo_5th_pct_proposed": after.get("monte_carlo_5th_pct"),
                    "post_rebalance_flags": json.dumps(flags),
                }
            )

        db_fields.update(
            {
                "regime_label": regime_curr.get("dominant_regime"),
                "regime_probabilities": json.dumps(
                    {
                        "Bull": regime_curr.get("Bull"),
                        "Neutral": regime_curr.get("Neutral"),
                        "Bear": regime_curr.get("Bear"),
                    }
                ),
                "risk_scalar": regime_curr.get("risk_scalar"),
                "tickers_analyzed": ",".join(prices.keys()),
                "sharpe_ratio": quant_ci.get("portfolio_sharpe"),
                "max_drawdown": _parse_pct(
                    quant_ci.get("portfolio_max_drawdown_pct")
                ),
                "var_95": quant_ci.get("portfolio_var_95_dollar"),
                "status": "complete",
            }
        )
        backtest_ci = adv_out.get("backtest_ci", {})
        if backtest_ci:
            db_fields["backtest_hrp_cagr"] = backtest_ci.get("hrp_strategy_cagr")
            db_fields["backtest_hrp_sharpe"] = backtest_ci.get("hrp_sharpe")
            db_fields["backtest_benchmark_cagr"] = backtest_ci.get("benchmark_cagr")

        # Use existing method, then patch full fields on latest row (method currently stores subset)
        db.save_analysis_run(db_fields)
        run_id = None
        with db.Session() as session:
            row = session.execute(select(AnalysisRun).order_by(AnalysisRun.id.desc()).limit(1)).scalar_one_or_none()
            if row is not None:
                for key, value in db_fields.items():
                    if hasattr(row, key):
                        setattr(row, key, value)
                session.commit()
                run_id = row.id
        logger.info("  Analysis run saved: id=%s", run_id)

        # Save ticker_metrics
        tickers = list(prices.keys())
        for ticker in tickers:
            garch = garch_output.get(ticker, {}) or {}
            risk = risk_output.get(ticker, {}) or {}
            hrp_w = (hrp_output.get("hrp_weights", {}) or {}).get(ticker)
            curr_w = (hrp_output.get("current_weights", {}) or {}).get(ticker)
            hrp_row = next(
                (r for r in (hrp_output.get("rebalancing_table", []) or []) if r.get("ticker") == ticker),
                {},
            )
            analyst = analyst_out.get(ticker, {}) or {}
            cvar = (
                (hrp_output.get("cvar_analysis", {}) or {})
                .get("marginal_cvar", {})
                .get(ticker)
            )
            conc_flag = (
                (hrp_output.get("cvar_analysis", {}) or {})
                .get("concentration_flags", {})
                .get(ticker, False)
            )
            trend_data = (trend_output.get(ticker, {}) if isinstance(trend_output, dict) else {}) or {}

            try:
                db.save_ticker_metrics(
                    {
                        "run_id": run_id,
                        "ticker": ticker,
                        "garch_annualized_vol": garch.get("annualized_vol"),
                        "garch_vol_regime": garch.get("vol_regime"),
                        "atr_14": risk.get("atr_14"),
                        "stop_loss": risk.get("stop_loss"),
                        "take_profit": risk.get("take_profit"),
                        "hrp_weight": hrp_w,
                        "current_weight": curr_w,
                        "weight_delta": hrp_row.get("delta"),
                        "rebalance_action": hrp_row.get("action"),
                        "analyst_target_price": analyst.get("target_price"),
                        "analyst_upside_pct": analyst.get("upside_pct"),
                        "analyst_signal": analyst.get("signal"),
                        "marginal_cvar_contribution": cvar,
                        "concentration_risk_flag": conc_flag,
                        "trend_direction": trend_data.get("trend_direction"),
                        "trend_slope_normalized": trend_data.get("trend_slope_normalized"),
                        "trend_slope_acceleration": trend_data.get("trend_slope_acceleration"),
                        "trend_deviation_std": trend_data.get("trend_deviation_std"),
                        "trend_uncertainty_pct": trend_data.get("trend_uncertainty_pct"),
                        "seasonal_component_pct": trend_data.get("seasonal_component_pct"),
                        "trend_confidence_score": trend_data.get("trend_confidence_score"),
                    }
                )
            except Exception as e:
                logger.warning("  ticker_metrics save failed for %s: %s", ticker, e)

        logger.info("  ticker_metrics saved: %d tickers", len(tickers))

        # Save regime_history
        try:
            db.save_regime_history(
                {
                    "date": date.today(),
                    "regime_label": regime_curr.get("dominant_regime"),
                    "bull_probability": regime_curr.get("Bull"),
                    "bear_probability": regime_curr.get("Bear"),
                    "crash_probability": regime_curr.get("Neutral"),
                    "risk_scalar": regime_curr.get("risk_scalar"),
                }
            )
            logger.info("  regime_history saved")
        except Exception as e:
            logger.warning("  regime_history save failed: %s", e)

        logger.info("DB saves complete (%.1fs)", time.time() - t0)
    except Exception as e:
        logger.error("STEP DB failed: %s", e, exc_info=True)
        raise


def step_generate_pdf(args) -> None:
    t0 = time.time()
    logger.info("=" * 50)
    logger.info("STEP PDF: GENERATING TEAR SHEET")
    logger.info("=" * 50)

    if args.skip_pdf:
        logger.info("--skip-pdf: skipping")
        return

    try:
        from reporting.pdf_generator import TearSheetGenerator

        cache_path = os.path.join(args.output_dir, "latest_run.json")
        output_path = os.path.join(args.output_dir, "tearsheet.pdf")
        gen = TearSheetGenerator(cache_path, output_path)
        gen.generate()
        logger.info("PDF saved to %s (%.1fs)", output_path, time.time() - t0)
    except Exception as e:
        logger.error("PDF generation failed: %s", e)
        logger.info("Dashboard still functional - PDF can be generated manually")


def main():
    parser = argparse.ArgumentParser(description="AI Portfolio Intelligence Pipeline")
    parser.add_argument(
        "--model",
        default=DEFAULT_LLM,
        choices=["anthropic", "anthropic-opus", "openai-gpt4o", "openai-gpt4o-mini"],
    )
    parser.add_argument("--skip-committee", action="store_true")
    parser.add_argument("--skip-pdf", action="store_true")
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip walk-forward backtest engine",
    )
    parser.add_argument("--price-period", default="2y", choices=["1y", "2y", "5y"])
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pipeline_start = time.time()
    logger.info("╔══════════════════════════════════╗")
    logger.info("║  AI PORTFOLIO INTELLIGENCE       ║")
    logger.info("║  Pipeline Starting               ║")
    logger.info("╚══════════════════════════════════╝")
    logger.info("Model:     %s", args.model)
    logger.info("Refresh:   %s", not args.no_refresh)
    logger.info("Committee: %s", not args.skip_committee)
    logger.info("Backtest:  %s", not args.skip_backtest)
    logger.info("PDF:       %s", not args.skip_pdf)

    db = DatabaseManager()
    db.init_db()

    try:
        data_out = step_load_data(args, db)
        quant_out = step_quant(data_out, db)
        adv_out = step_advanced_quant({**data_out}, args)
        ext_out = step_external_data({**data_out, **adv_out})
        committee_out = step_committee(data_out, quant_out, adv_out, ext_out, args)
        step_save_to_db(data_out, quant_out, adv_out, ext_out, committee_out, db)
        step_generate_pdf(args)

        total = time.time() - pipeline_start
        verdict = committee_out.get("verdict", {})
        flags = committee_out.get("validation_flags", [])

        logger.info("╔══════════════════════════════════╗")
        logger.info("║  PIPELINE COMPLETE               ║")
        logger.info("╚══════════════════════════════════╝")
        logger.info("Total runtime:  %.1fs", total)
        logger.info(
            "Risk Score:     %s/10",
            verdict.get("portfolio_risk_score", "N/A"),
        )
        logger.info(
            "Regime:         %s",
            adv_out["regime_current"].get("dominant_regime", "N/A"),
        )
        logger.info(
            "Risk Scalar:    %.3f",
            adv_out["regime_current"].get("risk_scalar", 0),
        )
        logger.info(
            "Validation:     %s",
            "PASSED" if not flags else f"{len(flags)} WARNING(S)",
        )
        logger.info("Cache:          %s", committee_out.get("cache_path", "—"))
        logger.info("PDF:            %s", os.path.join(args.output_dir, "tearsheet.pdf"))
        logger.info("Dashboard:      streamlit run reporting/dashboard.py")
    except Exception as e:
        logger.error("Pipeline FAILED at step: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
    