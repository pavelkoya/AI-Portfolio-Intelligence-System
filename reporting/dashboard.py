import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CUSTOM_CSS = """
<style>
  /* Dark professional theme */
  .stApp { background-color: #0e1117; }

  /* Metric card styling */
  [data-testid="metric-container"] {
      background: #1c2333;
      border: 1px solid #2d3748;
      border-radius: 8px;
      padding: 16px;
  }

  /* Section headers */
  .panel-header {
      font-size: 13px;
      font-weight: 600;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: #718096;
      margin-bottom: 16px;
      padding-bottom: 8px;
      border-bottom: 1px solid #2d3748;
  }

  /* Agent cards */
  .agent-card {
      background: #1c2333;
      border-radius: 8px;
      padding: 16px;
      border-left: 3px solid;
      margin-bottom: 12px;
  }
  .bull-card { border-left-color: #48bb78; }
  .bear-card { border-left-color: #fc8181; }
  .cro-card  { border-left-color: #63b3ed; }

  /* Verdict badges */
  .badge-add    { background:#276749; color:#9ae6b4;
                  padding:2px 8px; border-radius:4px;
                  font-size:11px; font-weight:700; }
  .badge-reduce { background:#742a2a; color:#feb2b2;
                  padding:2px 8px; border-radius:4px;
                  font-size:11px; font-weight:700; }
  .badge-hold   { background:#744210; color:#fbd38d;
                  padding:2px 8px; border-radius:4px;
                  font-size:11px; font-weight:700; }

  /* Hide Streamlit branding */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
</style>
"""


@st.cache_data(ttl=60)
def load_cache(path="outputs/latest_run.json"):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def _load_regime_history(days: int = 60):
    """Load regime history from SQLite."""
    try:
        from data.database import DatabaseManager
        import sqlite3
        db = DatabaseManager()
        conn = sqlite3.connect(db.db_path if hasattr(db, "db_path") else "portfolio_ai.db")
        df = pd.read_sql_query(
            f"""
            SELECT date, regime_label,
                   bull_probability,
                   bear_probability,
                   crash_probability,
                   risk_scalar
            FROM regime_history
            ORDER BY date DESC
            LIMIT {days}
            """,
            conn,
        )
        conn.close()
        return df.sort_values("date")
    except Exception:
        return pd.DataFrame()


def build_sidebar(data, timestamp):
    regime = data["inputs"]["regime"]
    risk_score = data["verdict"].get("portfolio_risk_score", "N/A")
    model = data.get("model_used", "Unknown")
    rs = regime.get("risk_scalar", 0)

    # Risk score color
    if isinstance(risk_score, int):
        if risk_score >= 8:
            score_color = "🔴"
        elif risk_score >= 5:
            score_color = "🟡"
        else:
            score_color = "🟢"
    else:
        score_color = "⚪"

    st.sidebar.title("📊 Portfolio Intelligence")
    st.sidebar.markdown("---")
    st.sidebar.metric("Portfolio Risk Score", f"{score_color} {risk_score}/10")
    st.sidebar.metric(
        "Market Regime",
        regime.get("dominant_regime", "Unknown"),
        delta=f"scalar: {rs:.3f}  "
        f"{'🔴 HIGH' if rs > 0.7 else '🟡 MED' if rs > 0.3 else '🟢 LOW'}",
        delta_color="inverse",
    )
    st.sidebar.metric(
        "Risk Scalar",
        f"{rs:.3f}",
        delta="BEAR REGIME" if rs > 0.7 else "ELEVATED" if rs > 0.3 else "NORMAL",
        delta_color="inverse",
    )
    st.sidebar.metric("AI Model", model)
    st.sidebar.markdown("---")

    # Timestamp
    try:
        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        friendly = dt.strftime("%b %d %Y %H:%M")
    except Exception:
        friendly = timestamp
    st.sidebar.caption(f"Last run: {friendly}")

    # Validation status
    flags = data.get("validation_flags", [])
    if flags:
        st.sidebar.error(f"⚠️ {len(flags)} validation warning(s)")
    else:
        st.sidebar.success("✅ Validation passed")

    st.sidebar.markdown("---")

    # Refresh button
    if st.sidebar.button("🔄 Refresh Cache"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    pdf_path = "outputs/tearsheet.pdf"
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            st.sidebar.download_button(
                "Download PDF Report",
                data=f.read(),
                file_name="portfolio_tearsheet.pdf",
                mime="application/pdf",
            )
    if st.sidebar.button("Generate PDF Report"):
        from reporting.pdf_generator import (
            TearSheetGenerator
        )

        with st.sidebar.spinner("Generating..."):
            gen = TearSheetGenerator()
            gen.generate()
        st.sidebar.success("PDF ready - click Download")
        st.rerun()


def panel_regime(inputs, regime):
    st.markdown(
        '<p class="panel-header">Market Regime Detection'
        " — Hidden Markov Model (3-state)</p>",
        unsafe_allow_html=True,
    )

    # Row 1: three metric cards
    col1, col2, col3, col4 = st.columns(4)
    bull = regime.get("Bull", 0)
    bear = regime.get("Bear", 0)
    neut = regime.get("Neutral", 0)
    rs = regime.get("risk_scalar", 0)

    with col1:
        st.metric("Bull Probability", f"{bull:.1%}", delta=None)
    with col2:
        st.metric("Neutral Probability", f"{neut:.1%}")
    with col3:
        st.metric("Bear Probability", f"{bear:.1%}", delta=None)
    with col4:
        st.metric(
            "Risk Scalar",
            f"{rs:.3f}",
            delta="HIGH" if rs > 0.7 else "NORMAL",
            delta_color="inverse",
        )

    st.markdown("---")

    # Row 2: gauge chart + regime explanation
    col_gauge, col_text = st.columns([1, 1])

    with col_gauge:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=rs,
                title={"text": "Risk Scalar", "font": {"color": "#e2e8f0", "size": 14}},
                number={"font": {"color": "#e2e8f0", "size": 36}},
                gauge={
                    "axis": {
                        "range": [0, 1],
                        "tickcolor": "#718096",
                        "tickfont": {"color": "#718096"},
                    },
                    "bar": {
                        "color": "#e53e3e" if rs > 0.7 else "#ecc94b" if rs > 0.3 else "#48bb78"
                    },
                    "bgcolor": "#1c2333",
                    "bordercolor": "#2d3748",
                    "steps": [
                        {"range": [0, 0.3], "color": "#1a3a2a"},
                        {"range": [0.3, 0.7], "color": "#3a3310"},
                        {"range": [0.7, 1.0], "color": "#3a1a1a"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.75,
                        "value": 0.7,
                    },
                },
            )
        )
        fig.update_layout(
            height=280,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_text:
        dominant = regime.get("dominant_regime", "Unknown")
        st.markdown("#### Current Assessment")

        if rs > 0.7:
            st.error(
                f"**{dominant} Regime** — Risk scalar "
                f"{rs:.3f} exceeds 0.7 threshold. "
                f"CRO mandate: minimum 20% cash allocation."
            )
        elif rs > 0.3:
            st.warning(
                f"**{dominant} Regime** — Elevated risk. "
                f"Monitor positions closely."
            )
        else:
            st.success(
                f"**{dominant} Regime** — Low risk scalar. "
                f"Normal positioning appropriate."
            )

        st.markdown("#### Regime Probabilities")
        prob_df = pd.DataFrame(
            {"State": ["Bull", "Neutral", "Bear"], "Probability": [bull, neut, bear]}
        )
        fig2 = px.bar(
            prob_df,
            x="State",
            y="Probability",
            color="State",
            color_discrete_map={"Bull": "#48bb78", "Neutral": "#ecc94b", "Bear": "#fc8181"},
            text_auto=".1%",
        )
        fig2.update_layout(
            height=200,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1c2333",
            font_color="#e2e8f0",
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            yaxis=dict(tickformat=".0%", gridcolor="#2d3748"),
            xaxis=dict(gridcolor="#2d3748"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Row 3: trend signals table
    st.markdown("#### Trend Signals (Linear Regression)")
    trend = inputs.get("trend_signals", {})
    low_conf = inputs.get("low_confidence_tickers", [])

    if trend:
        rows = []
        for ticker, t in trend.items():
            rows.append(
                {
                    "Ticker": ticker,
                    "Direction": t.get("trend_direction", "—"),
                    "Confidence": t.get("trend_confidence_score", 0),
                    "Slope (norm)": t.get("trend_slope_normalized", 0),
                    "Acceleration": t.get("trend_slope_acceleration", 0),
                    "Uncertainty %": t.get("trend_uncertainty_pct", 0),
                    "Seasonal": t.get("seasonal_component_pct", 0),
                    "Low Conf ⚠️": "⚠️" if ticker in low_conf else "",
                }
            )

        df = pd.DataFrame(rows)

        def color_direction(val):
            if val == "Up":
                return "color: #48bb78"
            if val == "Down":
                return "color: #fc8181"
            return ""

        def color_confidence(val):
            if val >= 0.6:
                return "color: #48bb78"
            if val >= 0.4:
                return "color: #ecc94b"
            return "color: #fc8181"

        styled = (
            df.style.format(
                {
                    "Confidence": "{:.2f}",
                    "Slope (norm)": "{:.6f}",
                    "Acceleration": "{:.6f}",
                    "Uncertainty %": "{:.2%}",
                    "Seasonal": "{:.2f}",
                }
            )
            .applymap(color_direction, subset=["Direction"])
            .applymap(color_confidence, subset=["Confidence"])
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.write("Coming soon")

    st.markdown("---")
    st.markdown("#### Regime History (Last 60 Days)")
    hist_df = _load_regime_history(60)

    if hist_df.empty:
        st.caption("No regime history yet — run the pipeline daily to build history.")
    else:
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Scatter(
                x=hist_df["date"],
                y=hist_df["risk_scalar"],
                name="Risk Scalar",
                line=dict(color="#63b3ed", width=2),
                fill="tozeroy",
                fillcolor="rgba(99,179,237,0.1)",
                mode="lines",
            )
        )

        for regime_name, color in {"Bull": "#48bb78", "Neutral": "#ecc94b", "Bear": "#fc8181"}.items():
            mask = hist_df["regime_label"] == regime_name
            if mask.any():
                fig_hist.add_trace(
                    go.Scatter(
                        x=hist_df[mask]["date"],
                        y=hist_df[mask]["risk_scalar"],
                        name=regime_name,
                        mode="markers",
                        marker=dict(color=color, size=8, symbol="circle"),
                    )
                )

        fig_hist.add_hline(
            y=0.7,
            line_dash="dash",
            line_color="#fc8181",
            annotation_text="Bear threshold (0.7)",
            annotation_position="right",
            annotation_font_color="#fc8181",
        )
        fig_hist.update_layout(
            height=250,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1c2333",
            font_color="#e2e8f0",
            yaxis=dict(title="Risk Scalar", range=[0, 1.05], gridcolor="#2d3748"),
            xaxis=dict(gridcolor="#2d3748"),
            legend=dict(bgcolor="#1c2333"),
            margin=dict(t=10, b=10, l=20, r=80),
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption(
            f"Showing {len(hist_df)} days of regime history. Run pipeline daily to extend the timeline."
        )


def panel_portfolio(inputs, post_reb, flags):
    st.markdown(
        '<p class="panel-header">Portfolio Health'
        " — HRP Optimization + Risk Metrics</p>",
        unsafe_allow_html=True,
    )

    risk_metrics = inputs.get("risk_metrics", {})
    optimization = inputs.get("optimization", {})
    hrp_table = optimization.get("hrp_rebalancing", [])
    max_dd = risk_metrics.get(
        "portfolio_max_drawdown_pct",
        risk_metrics.get("portfolio_max_drawdown", "N/A"),
    )
    var_val = risk_metrics.get("portfolio_var_95_dollar", 0)
    conc_ticker = optimization.get("concentration_risk_ticker") or "—"

    # Row 1: Key risk metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Sharpe Ratio", f"{risk_metrics.get('portfolio_sharpe', 0):.3f}")
    with m2:
        st.metric("Max Drawdown", max_dd)
    with m3:
        st.metric("VaR 95%", f"{var_val:.2f}")
    with m4:
        st.metric("Beta", f"{risk_metrics.get('portfolio_beta', 0):.3f}")
    with m5:
        st.metric(
            "Concentration",
            conc_ticker,
            delta="⚠️ Flagged" if conc_ticker != "—" else None,
            delta_color="inverse",
        )

    st.markdown("---")

    # Row 2: HRP rebalancing chart
    if hrp_table:
        st.markdown("#### HRP Rebalancing Recommendations")

        col_chart, col_table = st.columns([3, 2])

        tickers = [r["ticker"] for r in hrp_table]
        current = [r["current_weight"] * 100 for r in hrp_table]
        hrp_opt = [r["hrp_weight"] * 100 for r in hrp_table]
        deltas = [r["delta"] * 100 for r in hrp_table]
        actions = [r["action"] for r in hrp_table]

        with col_chart:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    name="Current Weight",
                    x=tickers,
                    y=current,
                    marker_color="#4a5568",
                    text=[f"{v:.1f}%" for v in current],
                    textposition="outside",
                    textfont=dict(color="#e2e8f0"),
                )
            )
            fig.add_trace(
                go.Bar(
                    name="HRP Optimal",
                    x=tickers,
                    y=hrp_opt,
                    marker_color="#63b3ed",
                    text=[f"{v:.1f}%" for v in hrp_opt],
                    textposition="outside",
                    textfont=dict(color="#e2e8f0"),
                )
            )
            fig.update_layout(
                barmode="group",
                height=350,
                paper_bgcolor="#0e1117",
                plot_bgcolor="#1c2333",
                font_color="#e2e8f0",
                legend=dict(bgcolor="#1c2333", bordercolor="#2d3748"),
                yaxis=dict(title="Weight %", gridcolor="#2d3748"),
                xaxis=dict(gridcolor="#2d3748"),
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.markdown("**Rebalancing Actions**")
            for row in sorted(hrp_table, key=lambda x: abs(x["delta"]), reverse=True):
                action = row["action"]
                ticker = row["ticker"]
                delta = row["delta"] * 100
                curr = row["current_weight"] * 100
                optimal = row["hrp_weight"] * 100

                badge_class = (
                    "badge-add"
                    if action == "ADD"
                    else "badge-reduce"
                    if action == "REDUCE"
                    else "badge-hold"
                )
                arrow = "↑" if delta > 0 else "↓"

                st.markdown(
                    f'<span class="{badge_class}">{action}</span> '
                    f"**{ticker}** "
                    f"{curr:.1f}% → {optimal:.1f}% "
                    f"({arrow}{abs(delta):.1f}%)",
                    unsafe_allow_html=True,
                )

    corr = inputs.get("optimization", {}).get("correlation", {})
    matrix_d = corr.get("matrix", {})
    tickers_c = corr.get("tickers", [])

    if matrix_d and tickers_c:
        st.markdown("#### Asset Correlation Matrix")
        z = []
        for t1 in tickers_c:
            row = [float(matrix_d.get(t1, {}).get(t2, 0) or 0) for t2 in tickers_c]
            z.append(row)

        fig_heat = go.Figure(
            go.Heatmap(
                z=z,
                x=tickers_c,
                y=tickers_c,
                colorscale=[
                    [0.0, "#1c2333"],
                    [0.5, "#2d3748"],
                    [0.75, "#63b3ed"],
                    [1.0, "#e53e3e"],
                ],
                zmid=0.5,
                zmin=-1,
                zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in z],
                texttemplate="%{text}",
                textfont={"size": 9},
                showscale=True,
            )
        )
        fig_heat.update_layout(
            height=400,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1c2333",
            font_color="#e2e8f0",
            yaxis=dict(autorange="reversed"),
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        pairs = corr.get("highly_correlated", [])
        if pairs:
            st.caption(
                "High correlation (r > 0.85): "
                + ", ".join(
                    f"{p[0]}↔{p[1]}" if isinstance(p, list) else str(p)
                    for p in pairs[:6]
                )
                + " — HRP assigns lower joint weight to these clusters."
            )

    st.markdown("---")

    # Row 3: Before/After validation
    st.markdown("#### Post-Rebalance Validation")

    if flags:
        for flag in flags:
            st.warning(flag)
    else:
        st.success(
            "✅ All validation checks passed — "
            "CRO recommendations are mathematically sound"
        )

    if post_reb:
        before = post_reb.get("before", {})
        after = post_reb.get("after", {})

        metrics = ["sharpe", "max_drawdown", "var_95", "beta", "monte_carlo_5th_pct"]
        labels = ["Sharpe Ratio", "Max Drawdown", "VaR 95%", "Beta", "Monte Carlo 5th Pct"]
        formats = ["{:.3f}", "{:.1%}", "{:.3f}", "{:.3f}", "{:.1%}"]

        rows = []
        for m, label, fmt in zip(metrics, labels, formats):
            b = before.get(m, 0)
            a = after.get(m, 0)
            d = a - b
            rows.append(
                {
                    "Metric": label,
                    "Before": fmt.format(b),
                    "After": fmt.format(a),
                    "Delta": d,
                    "Delta_display": fmt.format(d),
                }
            )

        df = pd.DataFrame(rows)

        c1, c2, c3, c4 = st.columns(4)
        headers = ["Metric", "Before", "After", "Δ"]
        for col, header in zip([c1, c2, c3, c4], headers):
            col.markdown(f"**{header}**")

        for _, row in df.iterrows():
            d = row["Delta"]
            m = row["Metric"]
            good = d > 0 if m in [
                "Sharpe Ratio",
                "Max Drawdown",
                "VaR 95%",
                "Monte Carlo 5th Pct"
            ] else d < 0
            color = "#48bb78" if good else "#fc8181"
            c1.write(row["Metric"])
            c2.write(row["Before"])
            c3.write(row["After"])
            c4.markdown(
                f'<span style="color:{color};font-weight:bold">{row["Delta_display"]}</span>',
                unsafe_allow_html=True,
            )


def panel_risk(inputs):
    st.markdown(
        '<p class="panel-header">Risk Levels'
        " — ATR Stop Loss & GARCH Volatility</p>",
        unsafe_allow_html=True,
    )

    per_ticker = inputs.get("per_ticker", {})
    trend = inputs.get("trend_signals", {})
    try:
        from data.database import DatabaseManager
        _db = DatabaseManager()
        _tickers = list(per_ticker.keys())
        prices_dict = {t: _db.get_prices(t, "2024-01-01", "2030-12-31") for t in _tickers}
    except Exception:
        prices_dict = {}

    rows = []
    for ticker, data in per_ticker.items():
        if data is None:
            continue
        t = trend.get(ticker, {})
        closes_14 = []
        if ticker in prices_dict:
            df_t = prices_dict[ticker]
            if df_t is not None and not df_t.empty and "close" in df_t.columns:
                closes_14 = df_t.tail(14)["close"].values.tolist()
        rows.append(
            {
                "Ticker": ticker,
                "Price": data.get("current_price") or data.get("stop_loss"),
                "Stop Loss": data.get("stop_loss"),
                "Take Profit": data.get("take_profit"),
                "RSI": data.get("rsi"),
                "RSI Signal": data.get("rsi_signal", "—"),
                "MACD": data.get("macd_signal", "—"),
                "Vol Regime": data.get("vol_regime", "—"),
                "GARCH Vol": data.get("garch_vol"),
                "Above SMA200": data.get("above_sma_200"),
                "Trend": t.get("trend_direction", "—"),
                "Conf": t.get("trend_confidence_score", 0),
                "14d": closes_14,
            }
        )

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "Stop Loss": st.column_config.NumberColumn("Stop Loss", format="$%.2f"),
                "Take Profit": st.column_config.NumberColumn("Take Profit", format="$%.2f"),
                "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                "GARCH Vol": st.column_config.NumberColumn("GARCH Vol", format="%.1f%%"),
                "Conf": st.column_config.NumberColumn("Conf", format="%.2f"),
                "14d": st.column_config.LineChartColumn(
                    "14d Price",
                    width="medium",
                    y_min=None,
                    y_max=None,
                ),
            },
        )

        st.caption(
            "Stop Loss = ATR×2.0 (×2.5 in Stress). "
            "Take Profit = 1.5:1 reward/risk ratio. "
            "GARCH Vol = annualized conditional volatility."
        )
    else:
        st.write("Coming soon")


def panel_backtest(inputs: dict):
    st.markdown(
        '<p class="panel-header">Walk-Forward Backtest — HRP+Regime vs Equal-Weight</p>',
        unsafe_allow_html=True,
    )

    bt = inputs.get("backtest_results", {})
    if not bt or not bt.get("n_periods"):
        st.info("No backtest data. Run: python main.py --price-period 5y")
        return

    note = bt.get("validity_note", "")
    if "VALID" in note:
        st.success(note)
    elif "ACCEPTABLE" in note:
        st.warning(note)
    else:
        st.error(note)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Periods", bt.get("n_periods", 0))
    m2.metric("HRP CAGR", f"{bt.get('hrp_strategy_cagr', 0):.1%}", delta=f"vs EQ: {bt.get('benchmark_cagr', 0):.1%}")
    m3.metric("HRP Sharpe", f"{bt.get('hrp_sharpe', 0):.3f}", delta=f"vs EQ: {bt.get('benchmark_sharpe', 0):.3f}")
    dd_r = bt.get("drawdown_reduction_pct", 0)
    m4.metric("Drawdown Reduction", f"{dd_r:.1%}", delta="capital preservation", delta_color="normal")
    m5.metric("Regime Timing Value", f"{bt.get('regime_timing_value', 0):+.2%}")

    eq_hrp = bt.get("equity_curve_hrp", [])
    eq_bench = bt.get("equity_curve_benchmark", [])

    if eq_hrp and eq_bench:
        st.markdown("#### Cumulative Return")
        dates_h = [p["date"] for p in eq_hrp]
        vals_h = [p["value"] * 100 for p in eq_hrp]
        dates_b = [p["date"] for p in eq_bench]
        vals_b = [p["value"] * 100 for p in eq_bench]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates_h + dates_b[::-1],
                y=vals_h + vals_b[::-1],
                fill="toself",
                fillcolor="rgba(99,179,237,0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dates_h,
                y=vals_h,
                name="HRP + Regime",
                line=dict(color="#63b3ed", width=2.5),
                mode="lines+markers",
                marker=dict(size=6),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dates_b,
                y=vals_b,
                name="Equal Weight",
                line=dict(color="#718096", width=2, dash="dash"),
                mode="lines+markers",
                marker=dict(size=6),
            )
        )
        fig.add_hline(y=0, line_color="#2d3748", line_width=1)
        fig.update_layout(
            height=350,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1c2333",
            font_color="#e2e8f0",
            yaxis=dict(title="Cumulative Return %", ticksuffix="%", gridcolor="#2d3748"),
            xaxis=dict(gridcolor="#2d3748"),
            legend=dict(bgcolor="#1c2333"),
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(bt.get("interpretation", ""))

        rolling = bt.get("rolling_sharpe", [])
        if rolling and len(rolling) > 1:
            st.markdown("#### Rolling Sharpe Per Period")
            rs_df = pd.DataFrame(rolling)
            fig2 = go.Figure()
            fig2.add_trace(
                go.Bar(
                    x=rs_df["date"],
                    y=rs_df["hrp"],
                    name="HRP Sharpe",
                    marker_color=["#48bb78" if v >= 0 else "#fc8181" for v in rs_df["hrp"]],
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=rs_df["date"],
                    y=rs_df["benchmark"],
                    name="EQ Sharpe",
                    line=dict(color="#718096", dash="dot"),
                    mode="lines",
                )
            )
            fig2.add_hline(y=0, line_color="#2d3748")
            fig2.update_layout(
                height=250,
                paper_bgcolor="#0e1117",
                plot_bgcolor="#1c2333",
                font_color="#e2e8f0",
                yaxis=dict(gridcolor="#2d3748"),
                xaxis=dict(gridcolor="#2d3748"),
                legend=dict(bgcolor="#1c2333"),
                margin=dict(t=20, b=10, l=20, r=20),
            )
            st.plotly_chart(fig2, use_container_width=True)


def panel_analyst(inputs):
    st.markdown(
        '<p class="panel-header">Analyst Context'
        " — Price Targets (yfinance)</p>",
        unsafe_allow_html=True,
    )

    per_ticker = inputs.get("per_ticker", {})

    rows = []
    for ticker, data in per_ticker.items():
        if data is None:
            continue
        target = data.get("analyst_target")
        # Get the raw float upside_pct from per_ticker
        # It may be stored as float (0.139) or
        # string ("13.9%") or None
        raw_upside = data.get("analyst_upside")

        # Normalize to float
        if isinstance(raw_upside, str):
            try:
                # Handle "13.9%" format
                upside_float = float(raw_upside.replace("%", "")) / 100
            except ValueError:
                upside_float = None
        elif isinstance(raw_upside, (int, float)):
            upside_float = float(raw_upside)
        else:
            upside_float = None
        signal = data.get("analyst_signal", "NO DATA")
        if target:
            rows.append(
                {
                    "Ticker": ticker,
                    "Target Price": target,
                    "Upside": upside_float,
                    "Upside Str": raw_upside if raw_upside else "N/A",
                    "Signal": signal,
                }
            )

    if rows:
        df = pd.DataFrame(rows)
        df["Upside_sort"] = df["Upside"].fillna(-999)
        df = df.sort_values("Upside_sort", ascending=False)
        df = df.drop(columns=["Upside_sort"])

        fig = px.bar(
            df,
            x="Ticker",
            y="Upside",
            color="Signal",
            color_discrete_map={
                "BUY": "#48bb78",
                "HOLD": "#ecc94b",
                "REDUCE": "#fc8181",
                "NO DATA": "#4a5568",
            },
            text=[f"{v:.1%}" if isinstance(v, float) else "N/A" for v in df["Upside"]],
            title="Analyst Upside vs Current Price",
        )
        fig.add_hline(
            y=0.10,
            line_dash="dash",
            line_color="#48bb78",
            annotation_text="BUY threshold (10%)",
            annotation_position="right",
        )
        fig.add_hline(
            y=-0.05,
            line_dash="dash",
            line_color="#fc8181",
            annotation_text="REDUCE threshold (-5%)",
            annotation_position="right",
        )
        fig.update_layout(
            height=350,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1c2333",
            font_color="#e2e8f0",
            yaxis=dict(tickformat=".0%", gridcolor="#2d3748", title="Analyst Upside"),
            xaxis=dict(gridcolor="#2d3748"),
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Target Price Summary")
        display_df = df[["Ticker", "Target Price", "Upside Str", "Signal"]].rename(
            columns={"Upside Str": "Upside %"}
        )

        def color_signal(val):
            if val == "BUY":
                return "color: #48bb78; font-weight:bold"
            if val == "REDUCE":
                return "color: #fc8181; font-weight:bold"
            if val == "HOLD":
                return "color: #ecc94b"
            return "color: #4a5568"

        styled = display_df.style.applymap(color_signal, subset=["Signal"]).format(
            {"Target Price": "${:.2f}"}
        )

        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.caption(
            "Source: yfinance analyst consensus. "
            "ETFs (VOO, VGT, QQQ etc.) have no "
            "analyst targets — shown as NO DATA."
        )
    else:
        st.info("No analyst data available.")


def panel_committee(verdict, post_reb, flags):
    st.markdown(
        '<p class="panel-header">AI Investment Committee'
        " — Bull · Bear · CRO</p>",
        unsafe_allow_html=True,
    )

    risk_score = verdict.get("portfolio_risk_score", "N/A")
    summary = verdict.get("executive_summary", "")
    positions = verdict.get("final_positions", [])
    bull_out = verdict.get("bull", {})
    bear_out = verdict.get("bear", {})
    cro_out = verdict.get("cro", {})

    # Row 1: Risk score + executive summary
    col_score, col_summary = st.columns([1, 3])

    with col_score:
        color = (
            "#e53e3e"
            if isinstance(risk_score, int) and risk_score >= 8
            else "#ecc94b"
            if isinstance(risk_score, int) and risk_score >= 5
            else "#48bb78"
        )
        st.markdown(
            f'<div style="text-align:center;padding:24px;background:#1c2333;'
            f'border-radius:8px;border:2px solid {color}">'
            f'<div style="font-size:11px;letter-spacing:2px;color:#718096;'
            f'text-transform:uppercase;margin-bottom:8px">Risk Score</div>'
            f'<div style="font-size:52px;font-weight:800;color:{color};line-height:1">{risk_score}</div>'
            f'<div style="font-size:24px;color:#4a5568">/10</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_summary:
        st.markdown("#### CRO Executive Summary")
        if summary:
            st.markdown(
                f'<div class="agent-card cro-card">{summary}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No summary available.")

    st.markdown("---")

    # Row 2: Final positions table
    st.markdown("#### Final Position Recommendations")

    if positions:
        order = {"REDUCE": 0, "ADD": 1, "HOLD": 2, "EXIT": 0}
        sorted_pos = sorted(
            positions,
            key=lambda x: (x.get("ticker") == "CASH", order.get(x.get("verdict", "HOLD"), 3)),
        )

        col1, col2 = st.columns([2, 3])

        with col1:
            for pos in sorted_pos:
                t = pos.get("ticker", "—")
                v = pos.get("verdict", "HOLD")
                w = pos.get("target_weight_pct", 0)
                badge = (
                    "badge-add"
                    if v == "ADD"
                    else "badge-reduce"
                    if v in ["REDUCE", "EXIT"]
                    else "badge-hold"
                )
                special = "🏦 " if t == "CASH" else ""
                st.markdown(
                    f'{special}<span class="{badge}">{v}</span> **{t}** → {w:.1f}%',
                    unsafe_allow_html=True,
                )

        with col2:
            pie_data = [p for p in sorted_pos if p.get("target_weight_pct", 0) > 0]
            if pie_data:
                labels = [p["ticker"] for p in pie_data]
                values = [p["target_weight_pct"] for p in pie_data]
                colors = [
                    "#4a5568"
                    if p["ticker"] == "CASH"
                    else "#48bb78"
                    if p["verdict"] == "ADD"
                    else "#fc8181"
                    if p["verdict"] in ["REDUCE", "EXIT"]
                    else "#ecc94b"
                    for p in pie_data
                ]
                fig = go.Figure(
                    go.Pie(
                        labels=labels,
                        values=values,
                        marker_colors=colors,
                        hole=0.4,
                        textinfo="label+percent",
                        textfont_color="#e2e8f0",
                    )
                )
                fig.update_layout(
                    height=320,
                    paper_bgcolor="#0e1117",
                    font_color="#e2e8f0",
                    legend=dict(bgcolor="#1c2333", font_color="#e2e8f0"),
                    margin=dict(t=20, b=20, l=20, r=20),
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Row 3: Agent debate
    st.markdown("#### Agent Debate")

    col_bull, col_bear = st.columns(2)

    with col_bull:
        st.markdown(
            '<div class="agent-card bull-card"><strong>🐂 Bull Agent</strong></div>',
            unsafe_allow_html=True,
        )
        bull_thesis = ""
        if isinstance(bull_out, dict):
            bull_thesis = bull_out.get("overall_bull_thesis", "")
            recs = bull_out.get("recommendations", [])
            for r in recs[:3]:
                t = r.get("ticker", "")
                w = r.get("proposed_weight_pct", 0)
                reason = r.get("reasoning", "")[:120]
                st.markdown(
                    f'<span class="badge-add">ADD</span> **{t}** → {w:.1f}%  \n'
                    f'<small style="color:#718096">{reason}...</small>',
                    unsafe_allow_html=True,
                )
        if bull_thesis:
            with st.expander("Full Bull Thesis"):
                st.write(bull_thesis)

    with col_bear:
        st.markdown(
            '<div class="agent-card bear-card"><strong>🐻 Bear Agent</strong></div>',
            unsafe_allow_html=True,
        )
        if isinstance(bear_out, dict):
            bear_thesis = bear_out.get("overall_bear_thesis", "")
            recs = bear_out.get("recommendations", [])
            for r in recs[:3]:
                t = r.get("ticker", "")
                w = r.get("proposed_weight_pct", 0)
                reason = r.get("reasoning", "")[:120]
                sl = r.get("stop_loss", "")
                st.markdown(
                    f'<span class="badge-reduce">REDUCE</span> **{t}** → {w:.1f}%'
                    f'{"  Stop: $"+str(sl) if sl else ""}  \n'
                    f'<small style="color:#718096">{reason}...</small>',
                    unsafe_allow_html=True,
                )
            if bear_thesis:
                with st.expander("Full Bear Thesis"):
                    st.write(bear_thesis)

    if isinstance(cro_out, dict):
        with st.expander("⚖️ Full CRO Assessment"):
            bull_assess = cro_out.get("bull_assessment", "")
            bear_assess = cro_out.get("bear_assessment", "")
            most_imp = cro_out.get("most_important_risk", "")
            cash_pct = cro_out.get("cash_allocation_pct", 0)
            if most_imp:
                st.error(f"**Most Important Risk:** {most_imp}")
            if cash_pct:
                st.warning(f"**Mandated Cash Allocation:** {cash_pct:.1f}%")
            if bull_assess:
                st.markdown(f"**Bull Assessment:** {bull_assess}")
            if bear_assess:
                st.markdown(f"**Bear Assessment:** {bear_assess}")


def main():
    st.set_page_config(
        page_title="Portfolio Intelligence",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    data = load_cache()

    if data is None:
        st.error("No analysis found.")
        st.info("Run: python main.py")
        st.stop()

    # Parse top-level objects
    inputs = data.get("inputs", {})
    verdict = data.get("verdict", {})
    regime = inputs.get("regime", {})
    post_reb = data.get("post_rebalance", {})
    flags = data.get("validation_flags", [])
    ts = data.get("timestamp", "unknown")

    # Sidebar (built first, always visible)
    build_sidebar(data, ts)

    # Six panels via st.tabs()
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["🌡️ Regime", "⚖️ Portfolio Health", "🛡️ Risk Levels", "🎯 Analyst Context", "🤖 Committee", "📈 Backtest"]
    )

    with tab1:
        panel_regime(inputs, regime)
    with tab2:
        panel_portfolio(inputs, post_reb, flags)
    with tab3:
        panel_risk(inputs)
    with tab4:
        panel_analyst(inputs)
    with tab5:
        panel_committee(verdict, post_reb, flags)
    with tab6:
        panel_backtest(inputs)


if __name__ == "__main__":
    main()
