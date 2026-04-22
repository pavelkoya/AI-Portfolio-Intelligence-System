import json
import logging

import anthropic

from config.settings import ANTHROPIC_API_KEY, DEFAULT_LLM, OPENAI_API_KEY
from ai.prompts import (
    BEAR_SYSTEM,
    BEAR_USER,
    BULL_SYSTEM,
    BULL_USER,
    CRO_SYSTEM,
    CRO_USER,
)

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "anthropic": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-5",
        "label": "Claude Sonnet 4.5",
    },
    "anthropic-opus": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-5",
        "label": "Claude Opus 4.5",
    },
    "openai-gpt4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "label": "GPT-4o",
    },
    "openai-gpt4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "label": "GPT-4o Mini (cheapest)",
    },
}


class InvestmentCommittee:
    def __init__(
        self,
        model_key: str = None,
        prices: dict = None,
        benchmark=None,
        portfolio: list = None,
    ):
        self.model_key = model_key or DEFAULT_LLM
        if self.model_key not in MODEL_REGISTRY:
            logger.warning(
                "Unknown model_key=%r; falling back to 'anthropic'", self.model_key
            )
            self.model_key = "anthropic"

        self.model_config = MODEL_REGISTRY[self.model_key]
        self.provider = self.model_config["provider"]

        if self.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "openai package not installed. Run: pip install openai"
                )
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        self.logger = logging.getLogger(__name__)
        self.bull_output = None
        self.bear_output = None
        self.cro_output = None
        self._prices = prices or {}
        self._benchmark = benchmark
        self._portfolio = portfolio or []
        self.proposed_weights = {}
        self.post_rebalance = None
        self.validation_flags = []
        self.logger.info(
            "Committee initialized with model: %s", self.model_config["label"]
        )

    def _call_agent(self, system: str, user: str, agent_name: str) -> dict:
        self.logger.info(
            "Calling %s via %s", agent_name, self.model_config.get("label")
        )

        max_tokens = 6000

        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model_config["model_id"],
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            raw = response.content[0].text.strip()
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_config["model_id"],
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            raw = response.choices[0].message.content.strip()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        try:
            clean = raw
            if "```" in clean:
                parts = clean.split("```")
                clean = parts[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            return json.loads(clean.strip())
        except json.JSONDecodeError as e:
            self.logger.error("%s: JSON truncated, attempting repair", agent_name)
            # Try to find last complete JSON object by truncating at last closing brace
            try:
                last_brace = raw.rfind("}")
                if last_brace > 0:
                    truncated = raw[: last_brace + 1]
                    # Strip markdown fences
                    if "```" in truncated:
                        parts = truncated.split("```")
                        truncated = parts[1]
                        if truncated.startswith("json"):
                            truncated = truncated[4:]
                    return json.loads(truncated.strip())
            except Exception:
                pass
            # Final fallback
            return {"error": str(e), "raw": raw[:500]}

    def available_models(self) -> list[dict]:
        options: list[dict] = []
        for key, cfg in MODEL_REGISTRY.items():
            provider = cfg.get("provider")
            available = True
            if provider == "openai":
                available = bool(OPENAI_AVAILABLE and OPENAI_API_KEY)
            options.append(
                {
                    "key": key,
                    "label": cfg.get("label"),
                    "provider": provider,
                    "available": bool(available),
                }
            )
        return options

    def run(self, committee_inputs_obj) -> dict:
        inputs_payload = (
            committee_inputs_obj.build_lean()
            if hasattr(committee_inputs_obj, "build_lean")
            else committee_inputs_obj
        )
        inputs_json = json.dumps(inputs_payload, indent=2, default=str)
        risk_scalar = (
            (inputs_payload.get("regime") or {}).get("current") or {}
        ).get("risk_scalar", 0.0)
        as_of = inputs_payload.get("as_of") or ""

        bull_user = BULL_USER.format(
            as_of=as_of, inputs_json=inputs_json, risk_scalar=float(risk_scalar)
        )
        bear_user = BEAR_USER.format(
            as_of=as_of, inputs_json=inputs_json, risk_scalar=float(risk_scalar)
        )

        self.bull_output = self._call_agent(BULL_SYSTEM, bull_user, "Bull Agent")
        self.bear_output = self._call_agent(BEAR_SYSTEM, bear_user, "Bear Agent")

        cro_user = CRO_USER.format(
            as_of=as_of,
            inputs_json=inputs_json,
            bull_output=json.dumps(self.bull_output, indent=2, default=str),
            bear_output=json.dumps(self.bear_output, indent=2, default=str),
            risk_scalar=float(risk_scalar),
        )
        self.cro_output = self._call_agent(CRO_SYSTEM, cro_user, "CRO")

        # Extract proposed weights from CRO final_positions
        self.proposed_weights = {}
        final_positions = self.cro_output.get("final_positions", []) if isinstance(self.cro_output, dict) else []

        for pos in final_positions:
            ticker = pos.get("ticker")
            weight = pos.get("target_weight_pct")
            if ticker and weight is not None:
                try:
                    self.proposed_weights[ticker] = float(weight)
                except (ValueError, TypeError):
                    pass

        self.logger.info(
            "Extracted %d proposed weights from CRO",
            len(self.proposed_weights)
        )

        from quant.post_rebalance_engine import (
            PostRebalanceEngine, validate_cro_claims
        )

        self.post_rebalance = None
        self.validation_flags = []

        if self.proposed_weights and len(
            self.proposed_weights
        ) >= 3:
            try:
                engine = PostRebalanceEngine(
                    prices=self._prices,
                    benchmark=self._benchmark,
                    current_portfolio=self._portfolio,
                    proposed_weights=self.proposed_weights
                )
                self.post_rebalance = engine.run()
                self.validation_flags = validate_cro_claims(
                    self.post_rebalance["before"],
                    self.post_rebalance["after"]
                )
                if self.validation_flags:
                    self.logger.warning(
                        "CRO validation: %d flags raised",
                        len(self.validation_flags)
                    )
                else:
                    self.logger.info(
                        "CRO validation: all checks passed"
                    )
            except Exception as e:
                self.logger.error(
                    "Post-rebalance engine failed: %s", e
                )

        verdict = {
            "bull": self.bull_output,
            "bear": self.bear_output,
            "cro": self.cro_output,
            "portfolio_risk_score": self.cro_output.get("portfolio_risk_score", "N/A")
            if isinstance(self.cro_output, dict)
            else "N/A",
            "executive_summary": self.cro_output.get(
                "executive_summary", "CRO output incomplete"
            )
            if isinstance(self.cro_output, dict)
            else "CRO output incomplete",
            "final_positions": self.cro_output.get("final_positions", [])
            if isinstance(self.cro_output, dict)
            else [],
            "cash_allocation_pct": self.cro_output.get("cash_allocation_pct", 0)
            if isinstance(self.cro_output, dict)
            else 0,
        }
        return verdict

    def get_db_fields(self, regime_output: dict = None) -> dict:
        regime_output = regime_output or {}
        current = (regime_output.get("current") or {}) if isinstance(regime_output, dict) else {}
        risk_scalar = current.get("risk_scalar")

        probs = {k: current.get(k) for k in ("Bull", "Neutral", "Bear") if k in current}
        dominant = current.get("dominant_regime")
        regime_label = dominant
        regime_probabilities = json.dumps(probs, default=str) if probs else None

        cro = self.cro_output or {}
        return {
            "regime_label": regime_label,
            "regime_probabilities": regime_probabilities,
            "risk_scalar": float(risk_scalar) if risk_scalar is not None else None,
            "bull_agent_output": json.dumps(self.bull_output, default=str)
            if self.bull_output is not None
            else None,
            "bear_agent_output": json.dumps(self.bear_output, default=str)
            if self.bear_output is not None
            else None,
            "cro_agent_output": json.dumps(self.cro_output, default=str)
            if self.cro_output is not None
            else None,
            "portfolio_risk_score": cro.get("portfolio_risk_score"),
            "executive_summary": cro.get("executive_summary"),
            "post_rebalance_before": json.dumps(
                self.post_rebalance["before"]
            ) if self.post_rebalance else None,
            "post_rebalance_after": json.dumps(
                self.post_rebalance["after"]
            ) if self.post_rebalance else None,
            "post_rebalance_flags": json.dumps(
                self.validation_flags
            ),
        }

    def save_cache(self,
                   committee_inputs: dict,
                   output_dir: str = "outputs") -> str:
        import os, json
        from datetime import datetime

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        cache = {
          "timestamp":        timestamp,
          "inputs":           committee_inputs,
          "verdict":          {
              "bull":               self.bull_output,
              "bear":               self.bear_output,
              "cro":                self.cro_output,
              "portfolio_risk_score":
                  self.cro_output.get("portfolio_risk_score")
                  if isinstance(self.cro_output, dict)
                  else None,
              "executive_summary":
                  self.cro_output.get("executive_summary")
                  if isinstance(self.cro_output, dict)
                  else None,
              "final_positions":
                  self.cro_output.get("final_positions", [])
                  if isinstance(self.cro_output, dict)
                  else [],
          },
          "proposed_weights": self.proposed_weights,
          "post_rebalance":   self.post_rebalance,
          "validation_flags": self.validation_flags,
          "model_used":       self.model_config["label"]
        }

        # Always overwrite latest run
        latest_path = os.path.join(
            output_dir, "latest_run.json"
        )
        with open(latest_path, "w") as f:
            json.dump(cache, f, indent=2, default=str)

        # Also save timestamped archive
        archive_path = os.path.join(
            output_dir, f"run_{timestamp}.json"
        )
        with open(archive_path, "w") as f:
            json.dump(cache, f, indent=2, default=str)

        self.logger.info(
            "Cache saved to %s", latest_path
        )
        return latest_path
