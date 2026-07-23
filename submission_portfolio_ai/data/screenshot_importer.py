from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path

from config.broker_maps import normalize_ticker
from config.settings import ANTHROPIC_API_KEY, GEMINI_API_KEY


PROMPT_TEXT = """
  You are analyzing a portfolio or brokerage
  screenshot. The interface may be in any language
  (English, Russian, German, etc.).

  First identify what each column represents by
  looking at headers or context clues.
  Common column meanings:
  - "Ticker" / "Symbol" / "Тикер" → ticker
  - "Qty" / "Quantity" / "Кол-во" / "Количество"
    → shares
  - "Entry Price" / "Avg Price" / "Средняя цена"
    → avg_price
  - "Price" / "Last" / "Цена" → current_price
  - "Value" / "Market Value" / "Стоимость"
    → market_value
  - "Share %" / "Weight" / "Доля" → weight_pct
  - "P&L %" / "Gain %" / "П/У %" → unrealized_pnl_pct

  Extract ALL visible stock/ETF positions.
  Strip exchange suffixes from tickers:
    "IAU.US" → "IAU"
    "ADS.EU" → "ADS"
    "META.US" → "META"

  Return ONLY a JSON array with no other text.
  Each element must have exactly these fields:
  {
    "ticker":             "AAPL",
    "shares":             10.0,
    "avg_price":          150.00,
    "current_price":      165.00,
    "market_value":       1650.00,
    "weight_pct":         12.5,
    "unrealized_pnl_pct": 10.0
  }

  Rules:
  - ticker: symbol only, NO exchange suffix
  - shares: quantity (0 if not visible)
  - avg_price: purchase/entry price (0 if not visible)
  - current_price: current market price (0 if unknown)
  - market_value: total current value in USD
  - weight_pct: portfolio % (0 if not visible)
  - unrealized_pnl_pct: gain/loss % (0 if not visible)
  - Skip section header rows, totals, cash rows
  - If value is ambiguous extract your best estimate
  - Return [] only if truly no positions visible

  Return JSON array only. No markdown. No preamble.
  """


class ScreenshotImporter:
    def __init__(self, backend: str = "anthropic"):
        self.backend = backend
        self.warnings = []
        self.logger = logging.getLogger(__name__)

    def _encode_image(self, filepath: str) -> tuple[str, str]:
        """
        Read image file and return (base64_str, media_type).
        """
        path = Path(filepath)
        suffix = path.suffix.lower()
        media_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        if suffix not in media_map:
            raise ValueError(f"Unsupported image type: {suffix}")

        media_type = media_map[suffix]
        img_bytes = path.read_bytes()
        encoded = base64.b64encode(img_bytes).decode("utf-8")
        return encoded, media_type

    def _call_anthropic(self, image_b64_list: list[tuple]) -> str:
        """
        Call Claude claude-sonnet-4-5 with vision.
        image_b64_list: list of (b64_str, media_type)
        """
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        content = []
        for b64, mtype in image_b64_list:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mtype,
                        "data": b64,
                    },
                }
            )
        content.append({"type": "text", "text": PROMPT_TEXT})

        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2000,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text.strip()

    def _call_gemini(self, image_b64_list: list[tuple]) -> str:
        """
        Call Google Gemini Flash with vision.
        Requires: pip install google-generativeai
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("pip install google-generativeai")

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-flash-latest")

        parts = []
        for b64, mtype in image_b64_list:
            img_bytes = base64.b64decode(b64)
            parts.append(
                {
                    "inline_data": {
                        "mime_type": mtype,
                        "data": base64.b64encode(img_bytes).decode(),
                    }
                }
            )
        parts.append(PROMPT_TEXT)

        response = model.generate_content(parts)
        return response.text.strip()

    def _parse_response(self, raw_text: str) -> list[dict]:
        """
        Parse JSON from AI response.
        Handles markdown code fences if present.
        """
        clean = raw_text or ""
        if "```" in clean:
            # Try to extract JSON array content from fenced output.
            m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", clean, flags=re.IGNORECASE)
            if m:
                clean = m.group(1)
            else:
                parts = clean.split("```")
                for part in parts:
                    p = part.strip()
                    if p.startswith("["):
                        clean = p
                        break
                    if p.startswith("json"):
                        clean = p[4:]
                        break

        try:
            data = json.loads(clean.strip())
            if not isinstance(data, list):
                return []
            return data
        except json.JSONDecodeError as e:
            self.logger.error("JSON parse failed: %s\nRaw: %s", e, raw_text[:300])
            return []

    def extract(self, image_paths: list[str]) -> list[dict]:
        """
        Main method. Takes list of image paths.
        Returns list of position dicts.
        """
        image_b64_list = []
        for path in image_paths:
            try:
                b64, mtype = self._encode_image(path)
                image_b64_list.append((b64, mtype))
            except Exception as e:
                self.warnings.append(f"Could not load {path}: {e}")

        if not image_b64_list:
            self.warnings.append("No valid images loaded")
            return []

        self.logger.info("Sending %d image(s) to %s", len(image_b64_list), self.backend)

        try:
            if self.backend == "anthropic":
                if not ANTHROPIC_API_KEY:
                    raise ValueError("ANTHROPIC_API_KEY not set")
                raw = self._call_anthropic(image_b64_list)
            elif self.backend == "gemini":
                if not GEMINI_API_KEY:
                    raise ValueError("GEMINI_API_KEY not set")
                raw = self._call_gemini(image_b64_list)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
        except Exception as e:
            self.warnings.append(f"Vision API call failed: {e}")
            return []

        positions = self._parse_response(raw)
        self.logger.info("Extracted %d raw positions", len(positions))

        seen = {}
        for pos in positions:
            ticker_raw = pos.get("ticker", "")
            if not ticker_raw:
                continue

            ticker, warn = normalize_ticker(ticker_raw)
            if warn:
                self.warnings.append(warn)
            pos["ticker"] = ticker

            # Deduplicate — keep higher market_value
            try:
                mv = float(pos.get("market_value", 0) or 0)
            except Exception:
                mv = 0.0

            if ticker in seen:
                try:
                    existing_mv = float(seen[ticker].get("market_value", 0) or 0)
                except Exception:
                    existing_mv = 0.0
                if mv > existing_mv:
                    seen[ticker] = pos
            else:
                seen[ticker] = pos

        normalized = list(seen.values())
        self.logger.info("Returning %d deduplicated positions", len(normalized))
        return normalized
