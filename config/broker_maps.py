from __future__ import annotations

from typing import Dict, List, Optional, Tuple


# Standard schema fields:
# - ticker
# - shares
# - avg_price
# - market_value
# - weight_pct
# - unrealized_pnl_pct
BROKER_MAPS: Dict[str, Dict[str, List[str]]] = {
    "robinhood": {
        "ticker": ["ticker", "symbol"],
        "shares": ["quantity", "shares"],
        "avg_price": ["average_buy_price", "avg_cost"],
        "market_value": ["equity", "market_value", "value"],
        "weight_pct": ["percentage", "weight_pct"],
        "unrealized_pnl_pct": ["percent_change", "unrealized_pnl_pct"],
    },
    "fidelity": {
        "ticker": ["Symbol", "symbol"],
        "shares": ["Quantity", "Shares"],
        "avg_price": ["Cost Basis Per Share", "Average Cost Basis"],
        "market_value": ["Current Value", "Market Value"],
        "weight_pct": ["Percent Of Account", "Weight"],
        "unrealized_pnl_pct": ["Gain/Loss Pct", "Unrealized Gain/Loss %"],
    },
    "schwab": {
        "ticker": ["Symbol"],
        "shares": ["Quantity"],
        "avg_price": ["Price Paid"],
        "market_value": ["Market Value"],
        "weight_pct": [],
        "unrealized_pnl_pct": ["Gain/Loss %"],
    },
    "ibkr": {
        "ticker": ["Financial Instrument", "Symbol"],
        "shares": ["Qty", "Position"],
        "avg_price": ["Avg Price", "Average Price"],
        "market_value": ["Value", "Market Value"],
        "weight_pct": [],
        "unrealized_pnl_pct": ["Unrealized P&L %"],
    },
    "freedom_broker": {
        "ticker": ["Тикер", "Инструмент", "ticker"],
        "shares": ["Кол-во", "Количество", "shares"],
        "avg_price": ["Средняя цена", "Цена покупки", "avg_price"],
        "market_value": ["Стоимость", "Рыночная стоимость", "market_value"],
        "weight_pct": ["Доля, %", "Доля", "weight_pct"],
        "unrealized_pnl_pct": ["Прибыль/убыток, %", "П/У %", "unrealized_pnl_pct"],
    },
    "freedom_broker_excel": {
        "ticker": ["Ticker"],
        "shares": ["Qty"],
        "avg_price": ["Entry Price"],
        "market_value": ["Residual value", "Value"],
        "weight_pct": ["Share (%)"],
        "unrealized_pnl_pct": ["For an entire period (%)"],
        "current_price": ["Price"],
    },
    "tinkoff": {
        "ticker": ["Тикер", "ISIN"],
        "shares": ["Количество"],
        "avg_price": ["Средняя цена покупки"],
        "market_value": ["Текущая стоимость"],
        "weight_pct": ["Доля в портфеле"],
        "unrealized_pnl_pct": ["Изменение, %"],
    },
}


# Ticker suffix normalization map
# Maps exchange suffixes to yfinance format
TICKER_SUFFIX_MAP: Dict[str, str] = {
    ".US": "",  # remove (already US)
    ".EU": ".DE",  # try German exchange
    ".DE": ".DE",  # keep
    ".L": ".L",  # London
    ".PA": ".PA",  # Paris
    ".AS": ".AS",  # Amsterdam
    ".MI": ".MI",  # Milan
    ".MC": ".MC",  # Madrid
    ".HK": ".HK",  # Hong Kong
    ".T": ".T",  # Tokyo
    ".SS": ".SS",  # Shanghai
    ".SZ": ".SZ",  # Shenzhen
    ".ME": ".ME",  # Moscow (historical)
    ".TO": ".TO",  # Toronto
    ".AX": ".AX",  # Australia
}


def normalize_ticker(raw_ticker: str) -> Tuple[str, Optional[str]]:
    """
    Normalize a ticker symbol to yfinance format.
    Returns (normalized_ticker, warning_or_None)
    """
    if not raw_ticker or not isinstance(raw_ticker, str):
        return raw_ticker, "Empty ticker"

    t = raw_ticker.strip().upper()

    for suffix, replacement in TICKER_SUFFIX_MAP.items():
        if t.endswith(suffix):
            normalized = t[: -len(suffix)] + replacement
            if replacement != suffix:
                return normalized, f"{raw_ticker} normalized to {normalized}"
            return normalized, None

    return t, None


def detect_broker(
    columns: List[str], maps: Optional[Dict[str, Dict[str, List[str]]]] = None
) -> Optional[str]:
    """
    Detect broker by checking how many column names
    match each broker map.
    Returns broker name or None if no match.
    Minimum 2 field matches required.
    """
    maps_to_use = maps or BROKER_MAPS
    scores: Dict[str, int] = {}
    col_set = {str(c).strip() for c in columns}

    for broker, mapping in maps_to_use.items():
        score = 0
        for _, variants in mapping.items():
            if any(v in col_set for v in variants):
                score += 1
        if score >= 2:
            scores[broker] = score

    if not scores:
        return None
    return max(scores, key=scores.get)


def load_user_maps() -> Dict:
    """Load user-saved custom broker mappings."""
    import json
    import os

    path = "outputs/user_broker_maps.json"
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_user_map(broker_name: str, mapping: Dict) -> None:
    """Save a user-defined broker mapping."""
    import json
    import os

    os.makedirs("outputs", exist_ok=True)
    path = "outputs/user_broker_maps.json"
    existing = load_user_maps()
    existing[broker_name] = mapping
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
