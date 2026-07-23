import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Ensure project root is importable when Streamlit runs from reporting/ context.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.broker_maps import BROKER_MAPS, load_user_maps, save_user_map
from data.csv_importer import CSVImporter
from data.screenshot_importer import ScreenshotImporter

st.set_page_config(
    page_title="Import Portfolio",
    page_icon="📥",
    layout="wide",
)

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


def _save_uploaded_to_tmp(uploaded_file) -> str:
    os.makedirs("/tmp", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = f"/tmp/{ts}_{uploaded_file.name}"
    with open(path, "wb") as f:
        f.write(uploaded_file.read())
    return path


def _editor_df_to_positions(editor_df: pd.DataFrame) -> list[dict]:
    positions = []
    for _, row in editor_df.iterrows():
        ticker = str(row.get("Ticker", "")).strip().upper()
        if not ticker:
            continue
        shares = float(row.get("Shares", 0) or 0)
        avg_p = float(row.get("Avg Price", 0) or 0)
        curr_p = float(row.get("Current Price", 0) or 0)
        mv = float(row.get("Value $", 0) or 0)
        w_pct = float(row.get("Weight %", 0) or 0)
        positions.append(
            {
                "ticker": ticker,
                "quantity": shares,
                "average_buy_price": avg_p,
                "current_price": curr_p,
                "market_value": mv,
                "weight_pct": w_pct,
                "equity_type": "equity",
                "unrealized_pnl": ((curr_p - avg_p) * shares if shares and avg_p else 0),
                "unrealized_pnl_pct": (((curr_p - avg_p) / avg_p * 100) if avg_p else 0),
            }
        )
    return positions


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("Portfolio Import")
    st.caption(
        "Import from CSV, Excel, screenshot, or manual entry. Edits are applied before "
        "running analysis."
    )

    # initialize state
    st.session_state.setdefault("imported_positions", None)
    st.session_state.setdefault("import_source", None)
    st.session_state.setdefault("import_warnings", [])
    st.session_state.setdefault("needs_manual_map", False)
    st.session_state.setdefault("raw_columns", [])
    st.session_state.setdefault("csv_importer", None)

    # no-op references to keep requested imports visible/available in this page
    _ = BROKER_MAPS
    _ = load_user_maps()
    _ = np.__name__
    _ = yf.__name__

    # ── STEP 1: Source selector ──
    st.markdown("### Step 1 — Choose Import Source")
    tab_csv, tab_screenshot, tab_manual = st.tabs(
        ["CSV / Excel", "Screenshot", "Manual Entry"]
    )

    positions = None
    warnings = []
    source = None

    # ── Tab 1: CSV / Excel ──
    with tab_csv:
        st.markdown("Upload one or two files. Supported formats: CSV, XLSX, XLS.")
        uploaded = st.file_uploader(
            "Portfolio file(s)",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="csv_upload",
        )
        if uploaded and st.button("Parse CSV", key="parse_csv"):
            with st.spinner("Detecting broker..."):
                importer = CSVImporter()
                try:
                    if len(uploaded) == 1:
                        tmp = _save_uploaded_to_tmp(uploaded[0])
                        importer.load_file(tmp)
                    else:
                        if len(uploaded) > 2:
                            st.warning("Using first two files only.")
                        tmp1 = _save_uploaded_to_tmp(uploaded[0])
                        tmp2 = _save_uploaded_to_tmp(uploaded[1])
                        importer.load_two_files(tmp1, tmp2)

                    col_map = importer.detect_and_map()

                    if col_map is None:
                        st.warning("Broker not recognized. Please map columns manually.")
                        st.session_state["needs_manual_map"] = True
                        st.session_state["raw_columns"] = importer.raw_df.columns.tolist()
                        st.session_state["csv_importer"] = importer
                    else:
                        st.success(f"Detected broker: **{importer.broker}**")
                        with st.spinner("Fetching current prices..."):
                            positions = importer.to_standard_schema()
                        warnings = importer.warnings
                        source = "csv"
                        st.session_state["imported_positions"] = positions
                        st.session_state["import_source"] = source
                        st.session_state["import_warnings"] = warnings
                        st.session_state["needs_manual_map"] = False

                except Exception as e:
                    st.error(f"Import failed: {e}")

        # Manual column mapping UI
        if st.session_state.get("needs_manual_map"):
            importer = st.session_state.get("csv_importer")
            raw_cols = st.session_state.get("raw_columns", [])

            st.markdown("#### Map Your Columns")
            st.caption("Select which column corresponds to each standard field.")

            standard_fields = [
                "ticker",
                "shares",
                "avg_price",
                "market_value",
                "weight_pct",
                "unrealized_pnl_pct",
            ]
            none_option = "— skip —"
            manual_map = {}

            cols = st.columns(2)
            for i, field in enumerate(standard_fields):
                with cols[i % 2]:
                    selected = st.selectbox(
                        f"{field}",
                        [none_option] + raw_cols,
                        key=f"map_{field}",
                    )
                    if selected != none_option:
                        manual_map[field] = selected

            broker_name = st.text_input(
                "Name this broker (for future use)",
                placeholder="e.g. my_broker",
            )

            if st.button("Apply Mapping"):
                if "ticker" not in manual_map:
                    st.error("Ticker column is required.")
                else:
                    if importer:
                        importer.apply_manual_map(manual_map)
                        if broker_name:
                            save_user_map(broker_name, manual_map)
                            st.success(f"Mapping saved as '{broker_name}'")
                        with st.spinner("Parsing with your mapping..."):
                            positions = importer.to_standard_schema()
                        warnings = importer.warnings
                        source = "csv_manual"
                        st.session_state["imported_positions"] = positions
                        st.session_state["import_source"] = source
                        st.session_state["import_warnings"] = warnings
                        st.session_state["needs_manual_map"] = False

    # ── Tab 2: Screenshot ──
    with tab_screenshot:
        st.markdown(
            "Upload one or more portfolio screenshots. AI will extract positions automatically."
        )

        backend = st.radio(
            "Vision AI backend",
            ["anthropic", "gemini"],
            horizontal=True,
            help=(
                "Anthropic: uses your existing API key. Gemini: requires GEMINI_API_KEY in .env"
            ),
        )

        screenshots = st.file_uploader(
            "Portfolio screenshot(s)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="screenshot_upload",
        )

        if screenshots and st.button("Extract Positions", key="extract_btn"):
            with st.spinner(f"Analyzing with {backend}..."):
                paths = []
                for img in screenshots:
                    paths.append(_save_uploaded_to_tmp(img))

                importer = ScreenshotImporter(backend=backend)
                try:
                    positions = importer.extract(paths)
                    warnings = importer.warnings
                    source = "screenshot"
                    st.session_state["imported_positions"] = positions
                    st.session_state["import_source"] = source
                    st.session_state["import_warnings"] = warnings
                    if positions:
                        st.success(f"Extracted {len(positions)} positions")
                    else:
                        st.warning("No positions found. Try a clearer screenshot.")
                except Exception as e:
                    st.error(f"Extraction failed: {e}")

    # ── Tab 3: Manual Entry ──
    with tab_manual:
        st.markdown("Start with an empty table and add positions manually.")
        if st.button("Start Empty Table"):
            st.session_state["imported_positions"] = []
            st.session_state["import_source"] = "manual"
            st.session_state["import_warnings"] = []
            positions = []
            source = "manual"

    # ── STEP 2: Editable table ──
    positions = st.session_state.get("imported_positions")
    warnings = st.session_state.get("import_warnings", [])
    source = st.session_state.get("import_source")

    if positions is not None:
        st.markdown("---")
        st.markdown("### Step 2 — Review & Edit")

        for w in warnings:
            st.warning(w)

        rows = []
        for p in positions:
            rows.append(
                {
                    "Ticker": p.get("ticker", p.get("Ticker", "")),
                    "Shares": float(p.get("quantity", p.get("Shares", 0)) or 0),
                    "Avg Price": float(p.get("average_buy_price", p.get("Avg Price", 0)) or 0),
                    "Current Price": float(p.get("current_price", p.get("Current Price", 0)) or 0),
                    "Value $": float(p.get("market_value", p.get("Value $", 0)) or 0),
                    "Weight %": float(p.get("weight_pct", p.get("Weight %", 0)) or 0),
                }
            )

        if not rows:
            rows = [
                {
                    "Ticker": "",
                    "Shares": 0.0,
                    "Avg Price": 0.0,
                    "Current Price": 0.0,
                    "Value $": 0.0,
                    "Weight %": 0.0,
                }
            ]

        df = pd.DataFrame(rows)

        total_w = df["Weight %"].sum()
        if total_w > 100.1:
            col_warn, col_btn = st.columns([3, 1])
            with col_warn:
                st.warning(f"Total weight: {total_w:.1f}% (exceeds 100%)")
            with col_btn:
                if st.button("Normalize Weights"):
                    if total_w > 0:
                        df["Weight %"] = df["Weight %"] / total_w * 100
                    st.rerun()

        col_title, col_refresh = st.columns([4, 1])
        with col_title:
            st.markdown("#### Edit Positions")
        with col_refresh:
            if st.button("🔄 Refresh Prices", use_container_width=True):
                tickers = df["Ticker"].tolist()
                tickers = [t for t in tickers if t]
                with st.spinner("Fetching prices..."):
                    try:
                        data = yf.download(
                            tickers,
                            period="1d",
                            progress=False,
                            auto_adjust=False,
                        )
                        if isinstance(data.columns, pd.MultiIndex):
                            close = data["Close"]
                        else:
                            close = data[["Close"]]
                            if len(tickers) == 1:
                                close.columns = [tickers[0]]
                        latest = close.iloc[-1].to_dict() if not close.empty else {}

                        for idx in range(len(df)):
                            t = df.at[idx, "Ticker"]
                            if t and t in latest:
                                p = float(latest[t] or 0)
                                df.at[idx, "Current Price"] = p
                                s = df.at[idx, "Shares"]
                                if s:
                                    df.at[idx, "Value $"] = round(s * p, 2)

                        total = df["Value $"].sum()
                        if total > 0:
                            df["Weight %"] = df["Value $"] / total * 100
                        st.session_state["imported_positions"] = _editor_df_to_positions(df)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Price fetch failed: {e}")

        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Shares": st.column_config.NumberColumn("Shares", format="%.4f"),
                "Avg Price": st.column_config.NumberColumn("Avg Price $", format="$%.2f"),
                "Current Price": st.column_config.NumberColumn(
                    "Current $",
                    format="$%.2f",
                    help="Auto-loaded from yfinance. Override manually if needed.",
                ),
                "Value $": st.column_config.NumberColumn("Value $", format="$%.2f"),
                "Weight %": st.column_config.NumberColumn("Weight %", format="%.2f%%"),
            },
            key="portfolio_editor",
        )

        # Auto-calculate Value and Weight after edit
        total_value_calc = 0.0
        for idx in range(len(edited_df)):
            shares = edited_df.at[idx, "Shares"]
            price = edited_df.at[idx, "Current Price"]
            value = edited_df.at[idx, "Value $"]
            avg_p = edited_df.at[idx, "Avg Price"]
            _ = avg_p

            # If Shares and Current Price set but Value $
            # is 0 or missing -> calculate Value
            if shares and price and (not value or value == 0):
                edited_df.at[idx, "Value $"] = shares * price
            # If Value and Current Price set but Shares
            # is 0 -> calculate Shares
            elif value and price and (not shares or shares == 0):
                edited_df.at[idx, "Shares"] = value / price

            total_value_calc += edited_df.at[idx, "Value $"] or 0

        # Recalculate Weight % for all rows
        if total_value_calc > 0:
            for idx in range(len(edited_df)):
                v = edited_df.at[idx, "Value $"] or 0
                edited_df.at[idx, "Weight %"] = v / total_value_calc * 100

        st.markdown("#### Add New Position")
        c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 2])

        with c1:
            new_ticker = st.text_input(
                "Ticker",
                placeholder="e.g. NVDA",
                key="new_ticker",
            ).upper().strip()
        with c2:
            new_mode = st.selectbox(
                "Specify by",
                ["Shares", "Total Value $", "Additional Weight %"],
                key="new_mode",
            )
        with c3:
            new_value = st.number_input(
                new_mode,
                min_value=0.0,
                step=1.0,
                key="new_value",
            )
        with c4:
            new_avg = st.number_input(
                "Avg Buy Price $",
                min_value=0.0,
                step=0.01,
                key="new_avg",
                help="Optional",
            )
        with c5:
            st.write("")  # spacer
            add_clicked = st.button(
                "➕ Add",
                key="add_ticker_btn",
                use_container_width=True,
            )

        if add_clicked and new_ticker:
            with st.spinner(f"Fetching {new_ticker}..."):
                try:
                    info = yf.Ticker(new_ticker).fast_info
                    curr_p = float(info.last_price or 0)
                except Exception:
                    curr_p = 0.0

            if new_mode == "Shares":
                shares_new = new_value
                value_new = shares_new * curr_p if curr_p else 0
            elif new_mode == "Total Value $":
                value_new = new_value
                shares_new = value_new / curr_p if curr_p else 0
            else:  # Additional Weight %
                cur_total = edited_df["Value $"].sum()
                if new_value >= 100:
                    st.error("Weight must be less than 100%")
                    value_new = 0
                    shares_new = 0
                else:
                    value_new = (
                        cur_total * new_value / (100 - new_value)
                        if new_value > 0
                        else 0
                    )
                    shares_new = value_new / curr_p if curr_p else 0

                if value_new > 0:
                    st.warning(
                        f"Adding {new_ticker} at {new_value:.1f}% will dilute all existing "
                        "positions proportionally. Confirm below."
                    )

            if value_new > 0 or shares_new > 0:
                new_row = pd.DataFrame(
                    [
                        {
                            "Ticker": new_ticker,
                            "Shares": round(shares_new, 4),
                            "Avg Price": new_avg or curr_p,
                            "Current Price": curr_p,
                            "Value $": round(value_new, 2),
                            "Weight %": 0.0,
                        }
                    ]
                )
                edited_df = pd.concat([edited_df, new_row], ignore_index=True)
                total_new = edited_df["Value $"].sum()
                if total_new > 0:
                    edited_df["Weight %"] = edited_df["Value $"] / total_new * 100
                st.session_state["imported_positions"] = _editor_df_to_positions(edited_df)
                st.rerun()

        new_total = edited_df["Weight %"].sum()
        if new_total > 100.1:
            st.warning(f"Total weight: {new_total:.1f}% — normalize before confirming.")

        # ── STEP 3: Confirm & Run ──
        st.markdown("---")
        st.markdown("### Step 3 — Confirm & Run")

        col_info, col_btn = st.columns([3, 1])
        with col_info:
            st.caption(
                f"Source: {source or 'unknown'} · {len(edited_df)} positions · "
                f"Total weight: {new_total:.1f}%"
            )

        with col_btn:
            run_clicked = st.button(
                "Confirm & Run Analysis",
                type="primary",
                use_container_width=True,
            )

        if run_clicked:
            final_positions = []
            total_value = edited_df["Value $"].sum()

            for _, row in edited_df.iterrows():
                ticker = str(row["Ticker"]).strip().upper()
                if not ticker:
                    continue
                mv = float(row["Value $"] or 0)
                shares = float(row["Shares"] or 0)
                avg_p = float(row["Avg Price"] or 0)
                curr_p = float(row["Current Price"] or 0)
                w_pct = float(row["Weight %"] or 0)

                if not w_pct and total_value > 0:
                    w_pct = (mv / total_value) * 100

                final_positions.append(
                    {
                        "ticker": ticker,
                        "quantity": shares,
                        "average_buy_price": avg_p,
                        "current_price": curr_p,
                        "market_value": mv,
                        "weight_pct": w_pct,
                        "equity_type": "equity",
                        "unrealized_pnl": ((curr_p - avg_p) * shares if shares and avg_p else 0),
                        "unrealized_pnl_pct": (((curr_p - avg_p) / avg_p * 100) if avg_p else 0),
                    }
                )

            os.makedirs("outputs", exist_ok=True)
            output = {
                "source": source,
                "imported_at": datetime.now().isoformat(),
                "positions": final_positions,
            }
            with open("outputs/imported_portfolio.json", "w") as f:
                json.dump(output, f, indent=2, default=str)

            st.success("Portfolio saved! Run the analysis pipeline with:")
            st.code("python main.py --skip-pdf")
            st.info(
                "The pipeline will automatically use this imported portfolio instead of "
                "the Robinhood API."
            )


if __name__ == "__main__":
    main()
