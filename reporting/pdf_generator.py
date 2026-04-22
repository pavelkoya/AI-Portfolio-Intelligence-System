import io
import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PAGE_W, PAGE_H = A4
MARGIN = 16 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

# Dark institutional palette
BG_DARK = HexColor("#0e1117")
BG_CARD = HexColor("#1c2333")
BG_HEADER = HexColor("#141925")
ACCENT_BLUE = HexColor("#63b3ed")
ACCENT_GREEN = HexColor("#48bb78")
ACCENT_RED = HexColor("#fc8181")
ACCENT_GOLD = HexColor("#ecc94b")
TEXT_WHITE = HexColor("#e2e8f0")
TEXT_MUTED = HexColor("#718096")
TEXT_DIM = HexColor("#4a5568")
BORDER = HexColor("#2d3748")
BULL_GREEN = HexColor("#276749")
BEAR_RED = HexColor("#742a2a")
HOLD_ORANGE = HexColor("#744210")

# Styles
STYLE_REPORT_TITLE = ParagraphStyle(
    "ReportTitle",
    fontSize=28,
    fontName="Helvetica-Bold",
    textColor=TEXT_WHITE,
    spaceAfter=4,
    leading=32,
)
STYLE_SECTION_LABEL = ParagraphStyle(
    "SectionLabel",
    fontSize=8,
    fontName="Helvetica-Bold",
    textColor=TEXT_MUTED,
    spaceAfter=6,
    leading=10,
    spaceBefore=14,
)
STYLE_H2 = ParagraphStyle(
    "H2",
    fontSize=14,
    fontName="Helvetica-Bold",
    textColor=TEXT_WHITE,
    spaceAfter=6,
    spaceBefore=10,
)
STYLE_H3 = ParagraphStyle(
    "H3",
    fontSize=11,
    fontName="Helvetica-Bold",
    textColor=ACCENT_BLUE,
    spaceAfter=4,
    spaceBefore=8,
)
STYLE_BODY = ParagraphStyle(
    "Body",
    fontSize=9,
    fontName="Helvetica",
    textColor=TEXT_WHITE,
    spaceAfter=4,
    leading=14,
)
STYLE_BODY_MUTED = ParagraphStyle(
    "BodyMuted",
    fontSize=8,
    fontName="Helvetica",
    textColor=TEXT_MUTED,
    spaceAfter=3,
    leading=12,
)
STYLE_CAPTION = ParagraphStyle(
    "Caption",
    fontSize=7,
    fontName="Helvetica",
    textColor=TEXT_DIM,
    spaceAfter=2,
    leading=10,
)
STYLE_METRIC_VALUE = ParagraphStyle(
    "MetricValue",
    fontSize=22,
    fontName="Helvetica-Bold",
    textColor=TEXT_WHITE,
    spaceAfter=0,
    leading=26,
    alignment=TA_CENTER,
)
STYLE_METRIC_LABEL = ParagraphStyle(
    "MetricLabel",
    fontSize=7,
    fontName="Helvetica",
    textColor=TEXT_MUTED,
    spaceAfter=2,
    alignment=TA_CENTER,
)
STYLE_VERDICT = ParagraphStyle(
    "Verdict",
    fontSize=8,
    fontName="Helvetica-Bold",
    textColor=TEXT_WHITE,
    spaceAfter=3,
    leading=12,
)


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def plotly_to_image(fig, width=480, height=280) -> Image:
    """Convert plotly figure to reportlab Image."""
    img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
    buf = io.BytesIO(img_bytes)
    # px -> mm conversion tuned to fit reportlab frame safely.
    mm_per_px = 0.264583
    return Image(buf, width=width * mm_per_px * mm, height=height * mm_per_px * mm)


def dark_figure_layout(fig, height=260):
    """Apply consistent dark theme to all charts."""
    fig.update_layout(
        height=height,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1c2333",
        font_color="#e2e8f0",
        font_size=10,
        margin=dict(t=30, b=20, l=30, r=20),
        legend=dict(
            bgcolor="#1c2333",
            bordercolor="#2d3748",
            font_color="#e2e8f0",
        ),
    )
    fig.update_xaxes(gridcolor="#2d3748", linecolor="#2d3748")
    fig.update_yaxes(gridcolor="#2d3748", linecolor="#2d3748")
    return fig


def metric_card_table(metrics: list[dict]) -> Table:
    """
    Build a row of metric cards as a Table.
    metrics: list of {"label": str, "value": str, "color": HexColor (optional)}
    Returns a Table that renders as card row.
    """
    n = max(len(metrics), 1)
    col_w = CONTENT_W / n

    header_row = []
    value_row = []

    for m in metrics:
        color = m.get("color", TEXT_WHITE)
        val_style = ParagraphStyle(
            "mv",
            fontSize=18,
            fontName="Helvetica-Bold",
            textColor=color,
            alignment=TA_CENTER,
            leading=22,
        )
        lbl_style = ParagraphStyle(
            "ml",
            fontSize=7,
            fontName="Helvetica",
            textColor=TEXT_MUTED,
            alignment=TA_CENTER,
        )
        value_row.append(Paragraph(str(m["value"]), val_style))
        header_row.append(Paragraph(str(m["label"]).upper(), lbl_style))

    tbl = Table([header_row, value_row], colWidths=[col_w] * n)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), BG_CARD),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [BG_CARD, BG_CARD]),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("LINEAFTER", (0, 0), (-2, -1), 0.5, BORDER),
                ("ROUNDEDCORNERS", [4]),
            ]
        )
    )
    return tbl


def verdict_badge(action: str) -> Paragraph:
    """
    Returns a colored badge Paragraph.
    action: "ADD" | "REDUCE" | "HOLD" | "EXIT"
    """
    colors = {
        "ADD": (BULL_GREEN, ACCENT_GREEN),
        "REDUCE": (BEAR_RED, ACCENT_RED),
        "HOLD": (HOLD_ORANGE, ACCENT_GOLD),
        "EXIT": (BEAR_RED, ACCENT_RED),
    }
    bg, fg = colors.get(str(action).upper(), (BG_CARD, TEXT_MUTED))
    style = ParagraphStyle(
        "badge",
        fontSize=7,
        fontName="Helvetica-Bold",
        textColor=fg,
        backColor=bg,
        borderPadding=(2, 5, 2, 5),
        alignment=TA_CENTER,
    )
    return Paragraph(str(action).upper(), style)


def section_divider(label: str) -> list:
    """Returns [HRFlowable, Paragraph] for section."""
    return [
        HRFlowable(width=CONTENT_W, thickness=0.5, color=BORDER, spaceAfter=6, spaceBefore=10),
        Paragraph(str(label).upper(), STYLE_SECTION_LABEL),
    ]


def page_background(canvas_obj, doc):
    """
    Called on every page via onPage callback.
    Draws dark background, header bar, page number, and footer line.
    """
    c = canvas_obj
    w, h = A4

    c.setFillColor(BG_DARK)
    c.rect(0, 0, w, h, fill=1, stroke=0)

    c.setFillColor(BG_HEADER)
    c.rect(0, h - 8 * mm, w, 8 * mm, fill=1, stroke=0)

    c.setFont("Helvetica-Bold", 7)
    c.setFillColor(ACCENT_BLUE)
    c.drawString(MARGIN, h - 5.5 * mm, "AI PORTFOLIO INTELLIGENCE")
    c.setFillColor(TEXT_MUTED)
    c.drawRightString(
        w - MARGIN, h - 5.5 * mm, "INSTITUTIONAL TEAR SHEET - CONFIDENTIAL"
    )

    c.setStrokeColor(BORDER)
    c.setLineWidth(0.5)
    c.line(MARGIN, 10 * mm, w - MARGIN, 10 * mm)

    c.setFont("Helvetica", 7)
    c.setFillColor(TEXT_MUTED)
    c.drawCentredString(w / 2, 6 * mm, f"Page {doc.page}")


def build_page1(data: dict) -> list:
    """
    Page 1: Portfolio Overview.
    """
    story = []
    inputs = data.get("inputs", {})
    verdict = data.get("verdict", {})
    regime = inputs.get("regime", {})
    risk_m = inputs.get("risk_metrics", {})
    mc = risk_m.get("monte_carlo", {})
    ts = data.get("timestamp", "")

    try:
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        date_str = dt.strftime("%B %d, %Y  %H:%M")
    except Exception:
        date_str = str(ts)

    risk_score = verdict.get("portfolio_risk_score", "N/A")
    dominant = regime.get("dominant_regime", "Unknown")
    rs = _safe_float(regime.get("risk_scalar"), 0)

    regime_color = ACCENT_RED if rs > 0.7 else ACCENT_GOLD if rs > 0.3 else ACCENT_GREEN
    score_color = (
        ACCENT_RED
        if isinstance(risk_score, int) and risk_score >= 8
        else ACCENT_GOLD
        if isinstance(risk_score, int) and risk_score >= 5
        else ACCENT_GREEN
    )

    title_data = [
        [
            Paragraph("PORTFOLIO INTELLIGENCE REPORT", STYLE_REPORT_TITLE),
            Table(
                [
                    [
                        Paragraph(
                            str(dominant).upper(),
                            ParagraphStyle(
                                "regime_badge",
                                fontSize=9,
                                fontName="Helvetica-Bold",
                                textColor=regime_color,
                                alignment=TA_CENTER,
                            ),
                        )
                    ],
                    [
                        Paragraph(
                            f"RISK SCORE  {risk_score}/10",
                            ParagraphStyle(
                                "rs_badge",
                                fontSize=16,
                                fontName="Helvetica-Bold",
                                textColor=score_color,
                                alignment=TA_CENTER,
                            ),
                        )
                    ],
                ],
                colWidths=[45 * mm],
            ),
        ]
    ]
    title_tbl = Table(title_data, colWidths=[CONTENT_W - 50 * mm, 50 * mm])
    title_tbl.setStyle(
        TableStyle(
            [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 0), (1, 0), "RIGHT")]
        )
    )
    story.append(title_tbl)

    story.append(
        Paragraph(
            f"Generated  {date_str}  -  Model: {data.get('model_used','')}",
            STYLE_BODY_MUTED,
        )
    )
    story.append(
        HRFlowable(width=CONTENT_W, thickness=1, color=ACCENT_BLUE, spaceAfter=10, spaceBefore=6)
    )

    port_val = _safe_float(inputs.get("portfolio_summary", {}).get("total_value"), 0)
    sharpe = _safe_float(risk_m.get("portfolio_sharpe"), 0)
    max_dd = risk_m.get("portfolio_max_drawdown_pct", "N/A")
    var_val = _safe_float(risk_m.get("portfolio_var_95_dollar"), 0)
    beta = _safe_float(risk_m.get("portfolio_beta"), 0)

    cards = metric_card_table(
        [
            {"label": "Portfolio Value", "value": f"${port_val:,.0f}"},
            {
                "label": "Sharpe Ratio",
                "value": f"{sharpe:.3f}",
                "color": ACCENT_GREEN if sharpe > 1.0 else ACCENT_GOLD if sharpe > 0.5 else ACCENT_RED,
            },
            {"label": "Max Drawdown", "value": str(max_dd), "color": ACCENT_RED},
            {"label": "VaR 95%", "value": f"{var_val:.3f}"},
            {
                "label": "Beta vs SPY",
                "value": f"{beta:.3f}",
                "color": ACCENT_GOLD if beta > 1 else TEXT_WHITE,
            },
        ]
    )
    story.append(cards)
    story.append(Spacer(1, 8))

    story += section_divider("MARKET REGIME")

    bull = _safe_float(regime.get("Bull"), 0)
    bear = _safe_float(regime.get("Bear"), 0)
    neut = _safe_float(regime.get("Neutral"), 0)

    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=rs,
            title={"text": "Risk Scalar", "font": {"color": "#e2e8f0", "size": 11}},
            number={"font": {"color": "#e2e8f0", "size": 28}, "valueformat": ".3f"},
            gauge={
                "axis": {"range": [0, 1], "tickcolor": "#718096", "tickfont": {"color": "#718096", "size": 8}},
                "bar": {"color": "#e53e3e" if rs > 0.7 else "#ecc94b" if rs > 0.3 else "#48bb78"},
                "bgcolor": "#1c2333",
                "bordercolor": "#2d3748",
                "steps": [
                    {"range": [0, 0.3], "color": "#1a3a2a"},
                    {"range": [0.3, 0.7], "color": "#3a3310"},
                    {"range": [0.7, 1.0], "color": "#3a1a1a"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.75, "value": 0.7},
            },
        )
    )
    dark_figure_layout(fig_gauge, height=220)

    fig_prob = go.Figure()
    fig_prob.add_trace(
        go.Bar(
            x=["Bull", "Neutral", "Bear"],
            y=[bull, neut, bear],
            marker_color=["#48bb78", "#ecc94b", "#fc8181"],
            text=[f"{v:.1%}" for v in [bull, neut, bear]],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=10),
        )
    )
    fig_prob.update_layout(yaxis=dict(tickformat=".0%", range=[0, 1.15]))
    dark_figure_layout(fig_prob, height=220)

    chart_row = Table(
        [[plotly_to_image(fig_gauge, 320, 220), plotly_to_image(fig_prob, 320, 220)]],
        colWidths=[CONTENT_W / 2, CONTENT_W / 2],
    )
    chart_row.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(chart_row)

    story += section_divider("REGIME HISTORY - LAST 60 DAYS")
    import sqlite3
    try:
        conn = sqlite3.connect("portfolio_ai.db")
        hist = pd.read_sql_query(
            "SELECT date, regime_label, risk_scalar "
            "FROM regime_history ORDER BY date "
            "LIMIT 60",
            conn,
        )
        conn.close()
    except Exception:
        hist = pd.DataFrame()

    if not hist.empty:
        hist = hist.sort_values("date")
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Scatter(
                x=hist["date"],
                y=hist["risk_scalar"],
                name="Risk Scalar",
                line=dict(color="#63b3ed", width=2),
                fill="tozeroy",
                fillcolor="rgba(99,179,237,0.1)",
                mode="lines",
            )
        )
        for regime_name, color in {"Bull": "#48bb78", "Neutral": "#ecc94b", "Bear": "#fc8181"}.items():
            mask = hist["regime_label"] == regime_name
            if mask.any():
                fig_hist.add_trace(
                    go.Scatter(
                        x=hist[mask]["date"],
                        y=hist[mask]["risk_scalar"],
                        name=regime_name,
                        mode="markers",
                        marker=dict(color=color, size=7, symbol="circle"),
                    )
                )
        fig_hist.add_hline(y=0.7, line_dash="dash", line_color="#fc8181")
        dark_figure_layout(fig_hist, height=220)
        fig_hist.update_layout(
            yaxis=dict(title="Risk Scalar", range=[0, 1.05]),
            legend=dict(orientation="h", y=1.08),
        )
        story.append(plotly_to_image(fig_hist, 680, 220))
    else:
        story.append(
            Paragraph(
                "No regime history yet - run the pipeline daily to build history.",
                STYLE_BODY_MUTED,
            )
        )

    story += section_divider("MONTE CARLO SIMULATION")
    if mc:
        mc_cards = metric_card_table(
            [
                {"label": "Initial Value", "value": f"${_safe_float(mc.get('initial_value'), 0):,.0f}"},
                {
                    "label": "5th Percentile (Worst)",
                    "value": f"${_safe_float(mc.get('percentile_5'), 0):,.0f}",
                    "color": ACCENT_RED,
                },
                {
                    "label": "50th Percentile (Base)",
                    "value": f"${_safe_float(mc.get('percentile_50'), 0):,.0f}",
                    "color": ACCENT_GOLD,
                },
                {
                    "label": "95th Percentile (Best)",
                    "value": f"${_safe_float(mc.get('percentile_95'), 0):,.0f}",
                    "color": ACCENT_GREEN,
                },
                {"label": "Expected Return", "value": f"{_safe_float(mc.get('expected_return_pct'), 0):.1%}"},
            ]
        )
        story.append(mc_cards)

    return story


def build_page2(data: dict) -> list:
    """
    Page 2: HRP Rebalancing + Post-Rebalance Validation.
    """
    story = []
    inputs = data.get("inputs", {})
    post_reb = data.get("post_rebalance", {})
    flags = data.get("validation_flags", [])
    optim = inputs.get("optimization", {})
    hrp_tbl = optim.get("hrp_rebalancing", []) or []

    story.append(Paragraph("PORTFOLIO OPTIMIZATION", STYLE_H2))
    story.append(HRFlowable(width=CONTENT_W, thickness=1, color=ACCENT_BLUE, spaceAfter=8, spaceBefore=2))

    story += section_divider("HIERARCHICAL RISK PARITY - WEIGHT COMPARISON")

    if hrp_tbl:
        tickers = [r.get("ticker", "N/A") for r in hrp_tbl]
        current = [_safe_float(r.get("current_weight"), 0) * 100 for r in hrp_tbl]
        hrp_opt = [_safe_float(r.get("hrp_weight"), 0) * 100 for r in hrp_tbl]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Current",
                x=tickers,
                y=current,
                marker_color="#4a5568",
                text=[f"{v:.1f}%" for v in current],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=9),
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
                textfont=dict(color="#e2e8f0", size=9),
            )
        )
        fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.1), yaxis=dict(title="Weight %"))
        dark_figure_layout(fig, height=240)
        story.append(plotly_to_image(fig, 680, 240))
        story.append(Spacer(1, 6))

    corr_data = data.get("inputs", {}).get("optimization", {}).get("correlation", {})
    matrix_dict = corr_data.get("matrix", {})
    tickers_list = corr_data.get("tickers", [])

    if matrix_dict and tickers_list:
        story += section_divider("ASSET CORRELATION MATRIX")
        z = []
        for t1 in tickers_list:
            row = []
            for t2 in tickers_list:
                val = matrix_dict.get(t1, {}).get(t2, 0)
                row.append(float(val or 0))
            z.append(row)

        fig_h = go.Figure(
            go.Heatmap(
                z=z,
                x=tickers_list,
                y=tickers_list,
                colorscale=[
                    [0.0, "#2d3748"],
                    [0.5, "#4a5568"],
                    [0.75, "#63b3ed"],
                    [1.0, "#e53e3e"],
                ],
                zmid=0.5,
                zmin=-1,
                zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in z],
                texttemplate="%{text}",
                textfont={"size": 7, "color": "#e2e8f0"},
                showscale=True,
                colorbar=dict(
                    tickfont=dict(color="#e2e8f0", size=8),
                    title=dict(text="r", font=dict(color="#e2e8f0")),
                ),
            )
        )
        dark_figure_layout(fig_h, height=300)
        fig_h.update_layout(
            xaxis=dict(tickfont=dict(size=8, color="#e2e8f0"), side="bottom"),
            yaxis=dict(tickfont=dict(size=8, color="#e2e8f0"), autorange="reversed"),
        )
        story.append(plotly_to_image(fig_h, 680, 300))

        pairs = corr_data.get("highly_correlated", [])
        if pairs:
            pair_strs = [f"{p[0]}↔{p[1]}" if isinstance(p, list) else str(p) for p in pairs[:5]]
            story.append(
                Paragraph(
                    "High correlation pairs (r > 0.85): "
                    + "  ·  ".join(pair_strs)
                    + "  — HRP assigns lower joint weight to these clusters.",
                    STYLE_CAPTION,
                )
            )

    story += section_divider("REBALANCING ACTIONS")

    if hrp_tbl:
        sorted_hrp = sorted(hrp_tbl, key=lambda x: abs(_safe_float(x.get("delta"), 0)), reverse=True)

        tbl_data = [
            [
                Paragraph("TICKER", STYLE_CAPTION),
                Paragraph("ACTION", STYLE_CAPTION),
                Paragraph("CURRENT", STYLE_CAPTION),
                Paragraph("HRP TARGET", STYLE_CAPTION),
                Paragraph("DELTA", STYLE_CAPTION),
            ]
        ]

        for row in sorted_hrp:
            action = str(row.get("action", "HOLD")).upper()
            delta = _safe_float(row.get("delta"), 0) * 100
            curr = _safe_float(row.get("current_weight"), 0) * 100
            target = _safe_float(row.get("hrp_weight"), 0) * 100
            arrow = "UP" if delta > 0 else "DOWN"

            delta_color = ACCENT_GREEN if action == "ADD" else ACCENT_RED if action in {"REDUCE", "EXIT"} else ACCENT_GOLD
            delta_style = ParagraphStyle("ds", fontSize=9, fontName="Helvetica-Bold", textColor=delta_color, alignment=TA_RIGHT)

            tbl_data.append(
                [
                    Paragraph(str(row.get("ticker", "N/A")), STYLE_VERDICT),
                    verdict_badge(action),
                    Paragraph(f"{curr:.1f}%", STYLE_BODY_MUTED),
                    Paragraph(f"{target:.1f}%", STYLE_BODY),
                    Paragraph(f"{arrow} {abs(delta):.1f}%", delta_style),
                ]
            )

        col_w = [CONTENT_W * 0.18, CONTENT_W * 0.18, CONTENT_W * 0.20, CONTENT_W * 0.20, CONTENT_W * 0.24]
        rebal_tbl = Table(tbl_data, colWidths=col_w)
        rebal_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), BG_HEADER),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [BG_CARD, BG_DARK]),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, ACCENT_BLUE),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.25, BORDER),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(rebal_tbl)

    story += section_divider("POST-REBALANCE VALIDATION")

    if flags:
        for flag in flags:
            story.append(
                Paragraph(
                    f"WARNING  {flag}",
                    ParagraphStyle("warn", fontSize=8, fontName="Helvetica", textColor=ACCENT_GOLD, spaceAfter=3),
                )
            )
    else:
        story.append(
            Paragraph(
                "PASS  All validation checks passed - CRO recommendations are mathematically sound",
                ParagraphStyle("pass", fontSize=9, fontName="Helvetica-Bold", textColor=ACCENT_GREEN, spaceAfter=6),
            )
        )

    if post_reb:
        before = post_reb.get("before", {}) or {}
        after = post_reb.get("after", {}) or {}

        metrics = [
            ("Sharpe Ratio", "sharpe", "{:.3f}", True),
            ("Max Drawdown", "max_drawdown", "{:.1%}", True),
            ("VaR 95%", "var_95", "{:.3f}", True),
            ("Beta", "beta", "{:.3f}", False),
            ("Monte Carlo 5th Pct", "monte_carlo_5th_pct", "{:.1%}", True),
        ]

        ba_data = [[Paragraph("METRIC", STYLE_CAPTION), Paragraph("BEFORE", STYLE_CAPTION), Paragraph("AFTER", STYLE_CAPTION), Paragraph("CHANGE", STYLE_CAPTION)]]

        for label, key, fmt, higher_better in metrics:
            b = _safe_float(before.get(key), 0)
            a = _safe_float(after.get(key), 0)
            d = a - b
            good = d > 0 if higher_better else d < 0
            d_color = ACCENT_GREEN if good else ACCENT_RED
            d_style = ParagraphStyle("ds2", fontSize=9, fontName="Helvetica-Bold", textColor=d_color, alignment=TA_RIGHT)
            ba_data.append(
                [
                    Paragraph(label, STYLE_BODY_MUTED),
                    Paragraph(fmt.format(b), STYLE_BODY),
                    Paragraph(fmt.format(a), STYLE_BODY),
                    Paragraph(fmt.format(d), d_style),
                ]
            )

        col_w2 = [CONTENT_W * 0.34, CONTENT_W * 0.22, CONTENT_W * 0.22, CONTENT_W * 0.22]
        ba_tbl = Table(ba_data, colWidths=col_w2)
        ba_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), BG_HEADER),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [BG_CARD, BG_DARK]),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, ACCENT_BLUE),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.25, BORDER),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(ba_tbl)

    return story


def build_page_backtest(data: dict) -> list:
    story = []
    backtest = data.get("inputs", {}).get("backtest_results") or data.get("backtest_ci", {})

    # Also check verdict inputs
    if not backtest:
        bt_raw = data.get("inputs", {})
        backtest = bt_raw.get("backtest_results", {})

    story.append(Paragraph("STRATEGY BACKTEST VALIDATION", STYLE_H2))
    story.append(HRFlowable(width=CONTENT_W, thickness=1, color=ACCENT_BLUE, spaceAfter=8, spaceBefore=2))
    story.append(
        Paragraph(
            "Walk-forward backtest: 252-day training window, 63-day test window. "
            "HRP + Regime-Adjusted vs Equal-Weight benchmark (same tickers, no SPY).",
            STYLE_BODY_MUTED,
        )
    )
    story.append(Spacer(1, 8))

    if not backtest or not backtest.get("n_periods"):
        story.append(
            Paragraph(
                "Backtest not available - run without --skip-backtest flag.",
                STYLE_BODY_MUTED,
            )
        )
        return story

    # Normalize key aliases (committee input uses hrp_cagr key)
    hrp_cagr = _safe_float(backtest.get("hrp_strategy_cagr", backtest.get("hrp_cagr", 0.0)), 0.0)
    bench_cagr = _safe_float(backtest.get("benchmark_cagr", 0.0), 0.0)
    hrp_sharpe = _safe_float(backtest.get("hrp_sharpe", 0.0), 0.0)
    bench_sharpe = _safe_float(backtest.get("benchmark_sharpe", 0.0), 0.0)
    regime_timing = _safe_float(backtest.get("regime_timing_value", 0.0), 0.0)

    # Summary metric cards
    n = backtest.get("n_periods", 0)
    cards = metric_card_table(
        [
            {"label": "Periods Tested", "value": str(n)},
            {
                "label": "HRP CAGR",
                "value": f"{hrp_cagr:.1%}",
                "color": ACCENT_GREEN if hrp_cagr > bench_cagr else ACCENT_RED,
            },
            {"label": "Benchmark CAGR", "value": f"{bench_cagr:.1%}"},
            {
                "label": "HRP Sharpe",
                "value": f"{hrp_sharpe:.3f}",
                "color": ACCENT_GREEN if hrp_sharpe > bench_sharpe else ACCENT_RED,
            },
            {
                "label": "Regime Timing Value",
                "value": f"{regime_timing:+.2%}",
                "color": ACCENT_GREEN if regime_timing > 0 else ACCENT_RED,
            },
        ]
    )
    story.append(cards)
    story.append(Spacer(1, 10))

    # Equity curve chart
    story += section_divider("CUMULATIVE RETURN - HRP vs EQUAL-WEIGHT")

    eq_hrp = backtest.get("equity_curve_hrp", [])
    eq_bench = backtest.get("equity_curve_benchmark", [])

    if eq_hrp and eq_bench:
        dates_hrp = [p["date"] for p in eq_hrp]
        values_hrp = [(_safe_float(p.get("value"), 0.0) * 100) for p in eq_hrp]
        dates_bench = [p["date"] for p in eq_bench]
        values_bench = [(_safe_float(p.get("value"), 0.0) * 100) for p in eq_bench]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates_hrp + dates_bench[::-1],
                y=values_hrp + values_bench[::-1],
                fill="toself",
                fillcolor="rgba(99,179,237,0.1)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dates_hrp,
                y=values_hrp,
                name="HRP + Regime",
                line=dict(color="#63b3ed", width=2),
                mode="lines+markers",
                marker=dict(size=4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dates_bench,
                y=values_bench,
                name="Equal Weight",
                line=dict(color="#718096", width=2, dash="dash"),
                mode="lines+markers",
                marker=dict(size=4),
            )
        )
        fig.add_hline(y=0, line_color="#2d3748", line_width=1)
        fig.update_layout(
            yaxis=dict(title="Cumulative Return %", tickformat=".1f", ticksuffix="%"),
            xaxis=dict(title="Test Period End Date"),
            legend=dict(orientation="h", y=1.1),
        )
        dark_figure_layout(fig, height=260)
        story.append(plotly_to_image(fig, 680, 260))
        story.append(Spacer(1, 8))
    else:
        story.append(
            Paragraph(
                "Equity curve unavailable — run with --price-period 5y for sufficient backtest periods.",
                STYLE_BODY_MUTED,
            )
        )

    # Per-period breakdown table
    story += section_divider("PERIOD-BY-PERIOD BREAKDOWN")

    periods = backtest.get("periods_detail", [])
    if periods:
        tbl_data = [
            [
                Paragraph("PERIOD", STYLE_CAPTION),
                Paragraph("TEST WINDOW", STYLE_CAPTION),
                Paragraph("REGIME", STYLE_CAPTION),
                Paragraph("HRP RET", STYLE_CAPTION),
                Paragraph("EQ RET", STYLE_CAPTION),
                Paragraph("ALPHA", STYLE_CAPTION),
            ]
        ]

        for p in periods:
            bear = bool(p.get("regime_bear", False))
            hrp_r = _safe_float(p.get("hrp_return", 0.0), 0.0)
            eq_r = _safe_float(p.get("eq_return", 0.0), 0.0)
            alpha = hrp_r - eq_r
            a_color = ACCENT_GREEN if alpha >= 0 else ACCENT_RED
            h_color = ACCENT_GREEN if hrp_r >= 0 else ACCENT_RED
            e_color = ACCENT_GREEN if eq_r >= 0 else ACCENT_RED

            def cp(val, color):
                return Paragraph(
                    val,
                    ParagraphStyle(
                        "cp",
                        fontSize=8,
                        fontName="Helvetica-Bold",
                        textColor=color,
                        alignment=TA_RIGHT,
                    ),
                )

            tbl_data.append(
                [
                    Paragraph(str(p.get("period", "-")), STYLE_BODY_MUTED),
                    Paragraph(
                        str(p.get("test_start", ""))[:10] + "->" + str(p.get("test_end", ""))[:10],
                        STYLE_CAPTION,
                    ),
                    Paragraph(
                        "BEAR" if bear else "BULL/NEU",
                        ParagraphStyle(
                            "rg",
                            fontSize=8,
                            fontName="Helvetica-Bold",
                            textColor=ACCENT_RED if bear else ACCENT_GREEN,
                            alignment=TA_CENTER,
                        ),
                    ),
                    cp(f"{hrp_r:+.2%}", h_color),
                    cp(f"{eq_r:+.2%}", e_color),
                    cp(f"{alpha:+.2%}", a_color),
                ]
            )

        col_w = [
            CONTENT_W * 0.08,
            CONTENT_W * 0.36,
            CONTENT_W * 0.14,
            CONTENT_W * 0.14,
            CONTENT_W * 0.14,
            CONTENT_W * 0.14,
        ]
        bt_tbl = Table(tbl_data, colWidths=col_w)
        bt_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), BG_HEADER),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [BG_CARD, BG_DARK]),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, ACCENT_BLUE),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.25, BORDER),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(bt_tbl)

    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            f"Walk-forward: {backtest.get('window_days',252)}-day train, {backtest.get('test_days',63)}-day test. "
            "HRP weights adjusted by regime risk_scalar. Benchmark = equal-weight across same tickers. "
            "No lookahead bias - each window fitted on training data only.",
            STYLE_CAPTION,
        )
    )

    return story


def build_page3(data: dict, prices: dict = None) -> list:
    """
    Page 3: Per-Ticker Risk Cards.
    """
    story = []
    inputs = data.get("inputs", {})
    per_ticker = inputs.get("per_ticker", {}) or {}
    trend = inputs.get("trend_signals", {}) or {}
    low_conf = inputs.get("low_confidence_tickers", []) or []

    story.append(Paragraph("PER-TICKER RISK CARDS", STYLE_H2))
    story.append(HRFlowable(width=CONTENT_W, thickness=1, color=ACCENT_BLUE, spaceAfter=8, spaceBefore=2))

    tickers = [t for t in per_ticker if per_ticker.get(t) is not None]
    card_w = (CONTENT_W - 6 * mm) / 2
    prices_ref = prices or data.get("_prices_ref", {})

    def _build_sparkline(ticker: str, prices_map: dict, days: int = 14):
        """
        Build a tiny plotly line chart for last N days of close prices.
        Returns plotly Image or None.
        """
        try:
            df = prices_map.get(ticker)
            if df is None or len(df) < days:
                return None

            recent = df.tail(days).copy()
            closes = recent["close"].values.tolist()
            if not closes:
                return None

            color = "#48bb78" if closes[-1] >= closes[0] else "#fc8181"
            fill_color = "rgba(72,187,120,0.15)" if color == "#48bb78" else "rgba(252,129,129,0.15)"

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    y=closes,
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    fill="tozeroy",
                    fillcolor=fill_color,
                )
            )
            fig.update_layout(
                height=40,
                width=120,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                margin=dict(t=2, b=2, l=2, r=2),
            )
            img_bytes = fig.to_image(format="png", width=120, height=40, scale=2)
            return Image(io.BytesIO(img_bytes), width=30 * mm, height=10 * mm)
        except Exception:
            return None

    def build_single_card(ticker, d, t_data):
        is_low_conf = ticker in low_conf
        t = t_data or {}

        vol_regime = d.get("vol_regime", "-")
        regime_color = ACCENT_RED if vol_regime == "Stress" else ACCENT_GOLD if vol_regime == "Elevated" else ACCENT_GREEN

        rsi_val = d.get("rsi")
        rsi_sig = d.get("rsi_signal", "-")
        rsi_color = ACCENT_RED if rsi_sig == "Overbought" else ACCENT_GREEN if rsi_sig == "Oversold" else TEXT_MUTED

        trend_dir = t.get("trend_direction", "-")
        trend_conf = _safe_float(t.get("trend_confidence_score"), 0)
        trend_color = ACCENT_GREEN if trend_dir == "Up" else ACCENT_RED if trend_dir == "Down" else TEXT_MUTED

        header_style = ParagraphStyle("ch", fontSize=12, fontName="Helvetica-Bold", textColor=TEXT_WHITE)
        macd = d.get("macd_signal", "-")
        above_200 = d.get("above_sma_200")
        sma_text = "Above SMA200 YES" if above_200 is True else "Below SMA200 NO" if above_200 is False else "-"
        sma_color = ACCENT_GREEN if above_200 is True else ACCENT_RED if above_200 is False else TEXT_MUTED

        analyst_tgt = d.get("analyst_target")
        analyst_sig = d.get("analyst_signal", "-")

        def field_row(label, value, v_color=None):
            lbl_p = Paragraph(label, ParagraphStyle("fl", fontSize=7, fontName="Helvetica", textColor=TEXT_MUTED))
            val_p = Paragraph(
                str(value),
                ParagraphStyle("fv", fontSize=8, fontName="Helvetica-Bold", textColor=v_color or TEXT_WHITE, alignment=TA_RIGHT),
            )
            return [lbl_p, val_p]

        card_data = [
            [
                Paragraph(str(ticker), header_style),
                Paragraph(
                    f"{vol_regime} VOL",
                    ParagraphStyle("vr", fontSize=7, fontName="Helvetica-Bold", textColor=regime_color, alignment=TA_RIGHT),
                ),
            ]
        ]

        if is_low_conf:
            card_data.append(
                [
                    Paragraph(
                        "WARNING  Low trend confidence",
                        ParagraphStyle("lc", fontSize=7, fontName="Helvetica", textColor=ACCENT_GOLD, spaceAfter=2),
                    ),
                    Paragraph("", STYLE_CAPTION),
                ]
            )

        stop = d.get("stop_loss")
        take = d.get("take_profit")
        card_data += [
            field_row("Stop Loss", f"${stop:.2f}" if stop else "-", ACCENT_RED),
            field_row("Take Profit", f"${take:.2f}" if take else "-", ACCENT_GREEN),
            field_row("RSI", f"{rsi_val:.1f}  {rsi_sig}" if rsi_val is not None else "-", rsi_color),
            field_row("MACD Signal", macd, ACCENT_GREEN if macd == "Bullish" else ACCENT_RED),
            field_row("Trend", f"{trend_dir}  conf:{trend_conf:.2f}", trend_color),
            field_row("SMA 200", sma_text, sma_color),
            field_row("GARCH Vol", f"{_safe_float(d.get('garch_vol'), 0):.1%}" if d.get("garch_vol") is not None else "-"),
            field_row(
                "Analyst Target",
                f"${_safe_float(analyst_tgt):.2f}" if analyst_tgt else "N/A",
                ACCENT_GREEN if analyst_sig == "BUY" else ACCENT_RED if analyst_sig == "REDUCE" else TEXT_MUTED,
            ),
        ]

        spark = _build_sparkline(ticker, prices_ref, days=14)
        if spark:
            card_data.append(
                [
                    Paragraph(
                        "14d price",
                        ParagraphStyle("sp", fontSize=7, fontName="Helvetica", textColor=TEXT_MUTED),
                    ),
                    spark,
                ]
            )

        card_tbl = Table(card_data, colWidths=[card_w * 0.55, card_w * 0.45])
        card_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), BG_CARD),
                    ("BACKGROUND", (0, 0), (-1, 0), BG_HEADER),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, ACCENT_BLUE),
                    ("LINEBELOW", (0, 1), (-1, -2), 0.25, BORDER),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("ROUNDEDCORNERS", [4]),
                    ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
                ]
            )
        )
        return card_tbl

    ticker_list = list(tickers)
    for i in range(0, len(ticker_list), 2):
        t1 = ticker_list[i]
        d1 = per_ticker[t1]
        tr1 = trend.get(t1, {})

        if i + 1 < len(ticker_list):
            t2 = ticker_list[i + 1]
            d2 = per_ticker[t2]
            tr2 = trend.get(t2, {})
            row = Table(
                [[build_single_card(t1, d1, tr1), Spacer(6 * mm, 1), build_single_card(t2, d2, tr2)]],
                colWidths=[card_w, 6 * mm, card_w],
            )
        else:
            row = Table(
                [[build_single_card(t1, d1, tr1), Spacer(6 * mm, 1), Spacer(card_w, 1)]],
                colWidths=[card_w, 6 * mm, card_w],
            )

        row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(row)
        story.append(Spacer(1, 5))

    return story


def build_page4(data: dict) -> list:
    """
    Page 4: Committee Reasoning + Final Verdict.
    """
    story = []
    verdict = data.get("verdict", {}) or {}
    bull_out = verdict.get("bull", {}) or {}
    bear_out = verdict.get("bear", {}) or {}
    cro_out = verdict.get("cro", {}) or {}
    positions = verdict.get("final_positions", []) or []
    summary = verdict.get("executive_summary", "")
    risk_score = verdict.get("portfolio_risk_score", "N/A")

    story.append(Paragraph("INVESTMENT COMMITTEE VERDICT", STYLE_H2))
    story.append(HRFlowable(width=CONTENT_W, thickness=1, color=ACCENT_BLUE, spaceAfter=8, spaceBefore=2))

    story += section_divider("CRO EXECUTIVE SUMMARY")

    score_color = (
        ACCENT_RED
        if isinstance(risk_score, int) and risk_score >= 8
        else ACCENT_GOLD
        if isinstance(risk_score, int) and risk_score >= 5
        else ACCENT_GREEN
    )

    summary_data = [
        [
            Table(
                [
                    [Paragraph("RISK SCORE", ParagraphStyle("rsl", fontSize=7, fontName="Helvetica-Bold", textColor=TEXT_MUTED, alignment=TA_CENTER))],
                    [Paragraph(f"{risk_score}", ParagraphStyle("rsv", fontSize=40, fontName="Helvetica-Bold", textColor=score_color, alignment=TA_CENTER, leading=44))],
                    [Paragraph("OUT OF 10", ParagraphStyle("rso", fontSize=7, fontName="Helvetica", textColor=TEXT_MUTED, alignment=TA_CENTER))],
                ],
                colWidths=[30 * mm],
            ),
            Paragraph(
                str(summary),
                ParagraphStyle("exec_sum", fontSize=9, fontName="Helvetica", textColor=TEXT_WHITE, leading=15, spaceAfter=4),
            ),
        ]
    ]
    summary_tbl = Table(summary_data, colWidths=[34 * mm, CONTENT_W - 38 * mm])
    summary_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), BG_CARD),
                ("BACKGROUND", (1, 0), (1, 0), BG_DARK),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (0, 0), 8),
                ("RIGHTPADDING", (0, 0), (0, 0), 8),
                ("LEFTPADDING", (1, 0), (1, 0), 10),
                ("BOX", (0, 0), (0, 0), 0.5, score_color),
                ("LINERIGHT", (0, 0), (0, 0), 1, score_color),
            ]
        )
    )
    story.append(summary_tbl)
    story.append(Spacer(1, 8))

    story += section_divider("AGENT DEBATE  -  BULL vs BEAR")

    def build_agent_table(agent_out, side):
        color = ACCENT_GREEN if side == "bull" else ACCENT_RED
        label = "BULL AGENT - Top Buys" if side == "bull" else "BEAR AGENT - Top Reduces"
        thesis_key = "overall_bull_thesis" if side == "bull" else "overall_bear_thesis"

        rows = [[Paragraph(label, ParagraphStyle("ah", fontSize=9, fontName="Helvetica-Bold", textColor=color))]]

        recs = agent_out.get("recommendations", []) if isinstance(agent_out, dict) else []
        for r in recs[:3]:
            ticker = r.get("ticker", "-")
            action = r.get("action", "-")
            weight = _safe_float(r.get("proposed_weight_pct"), 0)
            reason_full = r.get("reasoning", "") or ""
            reason = reason_full[:150]
            metrics = r.get("key_metrics_cited", [])

            rows.append([Paragraph(f"<b>{ticker}</b>  {action}  -> {weight:.1f}%", ParagraphStyle("rt", fontSize=9, fontName="Helvetica-Bold", textColor=TEXT_WHITE))])
            rows.append([Paragraph(reason + ("..." if len(reason_full) > 150 else ""), STYLE_BODY_MUTED)])
            if metrics:
                rows.append([Paragraph("Cited: " + ", ".join(str(m) for m in metrics[:4]), STYLE_CAPTION)])

        thesis = agent_out.get(thesis_key, "") if isinstance(agent_out, dict) else ""
        if thesis:
            rows.append(
                [
                    Paragraph(
                        thesis[:200] + ("..." if len(thesis) > 200 else ""),
                        ParagraphStyle("th", fontSize=8, fontName="Helvetica-Oblique", textColor=TEXT_MUTED, spaceBefore=4),
                    )
                ]
            )

        tbl = Table(rows, colWidths=[(CONTENT_W / 2) - 4 * mm])
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), BG_DARK),
                    ("BACKGROUND", (0, 0), (-1, 0), BG_CARD),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, color),
                    ("LINEBEFORE", (0, 0), (0, -1), 2, color),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        return tbl

    agent_row = Table(
        [[build_agent_table(bull_out, "bull"), Spacer(6 * mm, 1), build_agent_table(bear_out, "bear")]],
        colWidths=[(CONTENT_W / 2) - 3 * mm, 6 * mm, (CONTENT_W / 2) - 3 * mm],
    )
    agent_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(agent_row)

    if isinstance(cro_out, dict):
        most_imp = cro_out.get("most_important_risk", "")
        cash_pct = _safe_float(cro_out.get("cash_allocation_pct"), 0)
        if most_imp or cash_pct:
            story += section_divider("CRO DIRECTIVES")
            if most_imp:
                story.append(
                    Paragraph(
                        f"Most Important Risk:  {most_imp}",
                        ParagraphStyle("mir", fontSize=9, fontName="Helvetica-Bold", textColor=ACCENT_GOLD, spaceAfter=4),
                    )
                )
            if cash_pct:
                story.append(
                    Paragraph(
                        f"Mandated Cash Allocation:  {cash_pct:.1f}%  (risk_scalar > 0.7 rule triggered)",
                        ParagraphStyle("ca", fontSize=9, fontName="Helvetica-Bold", textColor=ACCENT_BLUE, spaceAfter=4),
                    )
                )

    story += section_divider("FINAL POSITION RECOMMENDATIONS")

    if positions:
        order = {"REDUCE": 0, "EXIT": 0, "ADD": 1, "HOLD": 2}
        sorted_pos = sorted(positions, key=lambda x: (x.get("ticker") == "CASH", order.get(x.get("verdict", "HOLD"), 3)))

        col1_pos = sorted_pos[: len(sorted_pos) // 2 + 1]
        col2_pos = sorted_pos[len(sorted_pos) // 2 + 1 :]

        def pos_table(pos_list):
            rows = [[Paragraph("TICKER", STYLE_CAPTION), Paragraph("VERDICT", STYLE_CAPTION), Paragraph("TARGET %", STYLE_CAPTION)]]
            for p in pos_list:
                t = p.get("ticker", "-")
                v = str(p.get("verdict", "HOLD")).upper()
                w = _safe_float(p.get("target_weight_pct"), 0)
                w_color = (
                    ACCENT_GREEN
                    if v == "ADD"
                    else ACCENT_RED
                    if v in ["REDUCE", "EXIT"]
                    else ACCENT_GOLD
                    if t == "CASH"
                    else TEXT_WHITE
                )
                rows.append(
                    [
                        Paragraph(("CASH " if t == "CASH" else "") + str(t), ParagraphStyle("pt", fontSize=9, fontName="Helvetica-Bold", textColor=w_color)),
                        verdict_badge(v),
                        Paragraph(f"{w:.1f}%", ParagraphStyle("pw", fontSize=9, fontName="Helvetica-Bold", textColor=w_color, alignment=TA_RIGHT)),
                    ]
                )
            col_w_p = [(CONTENT_W / 2 - 4 * mm) * 0.4, (CONTENT_W / 2 - 4 * mm) * 0.35, (CONTENT_W / 2 - 4 * mm) * 0.25]
            tbl = Table(rows, colWidths=col_w_p)
            tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), BG_HEADER),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [BG_CARD, BG_DARK]),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.5, ACCENT_BLUE),
                        ("LINEBELOW", (0, 1), (-1, -1), 0.25, BORDER),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ]
                )
            )
            return tbl

        pos_row = Table(
            [[pos_table(col1_pos), Spacer(6 * mm, 1), pos_table(col2_pos)]],
            colWidths=[(CONTENT_W / 2) - 3 * mm, 6 * mm, (CONTENT_W / 2) - 3 * mm],
        )
        pos_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(pos_row)

    return story


class TearSheetGenerator:
    def __init__(self, cache_path="outputs/latest_run.json", output_path="outputs/tearsheet.pdf"):
        self.cache_path = cache_path
        self.output_path = output_path
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> dict:
        with open(self.cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate(self) -> str:
        data = self.load_data()
        try:
            from data.database import DatabaseManager
            db = DatabaseManager()
            tickers = list(data.get("inputs", {}).get("per_ticker", {}).keys())
            prices_ref = {}
            for t in tickers:
                df = db.get_prices(t, "2024-01-01", "2030-12-31")
                if not df.empty:
                    prices_ref[t] = df
        except Exception:
            prices_ref = {}

        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=A4,
            leftMargin=MARGIN,
            rightMargin=MARGIN,
            topMargin=12 * mm,
            bottomMargin=14 * mm,
            title="Portfolio Intelligence Tear Sheet",
            author="AI Portfolio System",
            subject="Institutional Portfolio Analysis",
        )

        story = []
        story += build_page1(data)
        story.append(PageBreak())
        story += build_page2(data)
        story.append(PageBreak())
        story += build_page_backtest(data)
        story.append(PageBreak())
        story += build_page3(data, prices=prices_ref)
        story.append(PageBreak())
        story += build_page4(data)

        doc.build(story, onFirstPage=page_background, onLaterPages=page_background)

        self.logger.info("Tear sheet saved to %s", self.output_path)
        return self.output_path


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    cache = sys.argv[1] if len(sys.argv) > 1 else "outputs/latest_run.json"
    output = sys.argv[2] if len(sys.argv) > 2 else "outputs/tearsheet.pdf"
    gen = TearSheetGenerator(cache, output)
    path = gen.generate()
    print(f"PDF generated: {path}")
