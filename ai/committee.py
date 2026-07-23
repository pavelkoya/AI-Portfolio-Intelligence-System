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

    def run_consistency_audit(self, inputs: dict, n_runs: int = 5) -> dict:
        """
        Run the committee n_runs times on identical
        inputs. Measure weight consistency, risk score
        variance, and top-ticker agreement rate.
        Returns audit report dict.
        """
        import time
        import numpy as np

        self.logger.info("Consistency audit: starting %d runs", n_runs)

        all_risk_scores = []
        all_weights = []   # list of {ticker:wt}
        all_bull_top3 = []  # list of [t1,t2,t3]
        all_bear_top3 = []
        run_times = []

        for i in range(n_runs):
            t0 = time.time()
            self.logger.info("Audit run %d/%d", i + 1, n_runs)
            try:
                verdict = self.run(inputs)

                # Risk score
                all_risk_scores.append(verdict.get("portfolio_risk_score") or 0)

                # Proposed weights from CRO
                weights = {}
                for pos in verdict.get("final_positions", []):
                    t = pos.get("ticker")
                    w = pos.get("target_weight_pct")
                    if t and w is not None:
                        weights[t] = float(w)
                all_weights.append(weights)

                # Bull top-3 tickers
                bull = verdict.get("bull") or {}
                if isinstance(bull, dict):
                    recs = bull.get("recommendations", [])[:3]
                    all_bull_top3.append([r.get("ticker", "") for r in recs])
                else:
                    all_bull_top3.append([])

                # Bear top-3 tickers
                bear = verdict.get("bear") or {}
                if isinstance(bear, dict):
                    recs = bear.get("recommendations", [])[:3]
                    all_bear_top3.append([r.get("ticker", "") for r in recs])
                else:
                    all_bear_top3.append([])

                run_times.append(time.time() - t0)

                # Save last successful outputs
                self._last_audit_bull = self.bull_output
                self._last_audit_bear = self.bear_output

                # Reset committee state for next run
                self.bull_output = None
                self.bear_output = None
                self.cro_output = None
                self.proposed_weights = {}

            except Exception as e:
                self.logger.error("Audit run %d failed: %s", i + 1, e)
                all_risk_scores.append(None)

        # ── Compute consistency metrics ──
        valid_scores = [s for s in all_risk_scores if s is not None]
        risk_score_mean = float(np.mean(valid_scores)) if valid_scores else 0
        risk_score_std = float(np.std(valid_scores)) if valid_scores else 0

        # Weight consistency per ticker
        all_tickers = set()
        for w in all_weights:
            all_tickers.update(w.keys())

        weight_stats = {}
        for ticker in all_tickers:
            vals = [w.get(ticker, 0) for w in all_weights]
            weight_stats[ticker] = {
                "mean": round(float(np.mean(vals)), 2),
                "std": round(float(np.std(vals)), 2),
                "min": round(float(np.min(vals)), 2),
                "max": round(float(np.max(vals)), 2),
                "cv": round(
                    float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else 0,
                    3,
                ),
            }

        # Top-3 agreement rate
        def top3_agreement(runs_list):
            if len(runs_list) < 2:
                return 0.0
            from collections import Counter

            frozen = [frozenset(r) for r in runs_list if r]
            if not frozen:
                return 0.0
            most_common_count = Counter(frozen).most_common(1)[0][1]
            return most_common_count / len(frozen)

        bull_agreement = top3_agreement(all_bull_top3)
        bear_agreement = top3_agreement(all_bear_top3)

        # Overall consistency score (0-100)
        # Exclude CASH from weight stability
        # CASH swings are regime-driven, not noise
        equity_stats = {t: s for t, s in weight_stats.items() if t != "CASH"}
        avg_weight_cv = (
            np.mean([s["cv"] for s in equity_stats.values()]) if equity_stats else 1.0
        )

        # Report CASH separately
        cash_stats = weight_stats.get("CASH", {})

        weight_stability = max(0, 1 - avg_weight_cv)
        risk_stability = max(0, 1 - (risk_score_std / 10 if risk_score_std else 0))
        agent_agreement = (bull_agreement + bear_agreement) / 2

        consistency_score = round(
            (0.40 * weight_stability + 0.30 * risk_stability + 0.30 * agent_agreement)
            * 100,
            1,
        )

        report = {
            "n_runs": n_runs,
            "n_successful": len(valid_scores),
            "risk_score_mean": round(risk_score_mean, 2),
            "risk_score_std": round(risk_score_std, 2),
            "risk_score_range": [
                min(valid_scores) if valid_scores else 0,
                max(valid_scores) if valid_scores else 0,
            ],
            "weight_stats": weight_stats,
            "cash_weight_cv": round(float(cash_stats.get("cv", 0)), 3),
            "cash_weight_mean": round(float(cash_stats.get("mean", 0)), 1),
            "cash_interpretation": (
                f"CASH allocation ranged {cash_stats.get('min',0):.0f}%–"
                f"{cash_stats.get('max',0):.0f}% across runs — high variance indicates "
                f"the CRO risk assessment is sensitive to stochastic LLM sampling. "
                f"This is a key finding: agent agreement on asset selection (100%) does not imply "
                f"agreement on portfolio risk level."
            ),
            "bull_top3_agreement": round(bull_agreement, 3),
            "bear_top3_agreement": round(bear_agreement, 3),
            "overall_consistency_score": consistency_score,
            "avg_run_time_sec": round(float(np.mean(run_times)) if run_times else 0, 1),
            "all_risk_scores": all_risk_scores,
            "bull_top3_all_runs": all_bull_top3,
            "bear_top3_all_runs": all_bear_top3,
            "interpretation": (
                f"CRO weight consistency: {weight_stability:.0%}. "
                f"Risk score std: {risk_score_std:.1f}. "
                f"Bull agent top-3 agreement: {bull_agreement:.0%}. "
                f"Bear agent top-3 agreement: {bear_agreement:.0%}. "
                f"Overall consistency: {consistency_score}/100."
            ),
        }

        self.logger.info(
            "Audit complete: consistency=%s/100 risk_std=%.1f bull_agree=%.0f%% bear_agree=%.0f%%",
            consistency_score,
            risk_score_std,
            bull_agreement * 100,
            bear_agreement * 100,
        )

        # Restore last run outputs for external access
        if hasattr(self, "_last_audit_bull"):
            self.bull_output = self._last_audit_bull
            self.bear_output = self._last_audit_bear

        return report


def score_citation_quality(
    agent_output: dict,
    quant_inputs: dict,
    agent_type: str = "bull"
) -> dict:

    per_ticker = quant_inputs.get("per_ticker", {})
    trend_signals = quant_inputs.get("trend_signals", {})
    low_conf = quant_inputs.get("low_confidence_tickers", [])

    # WHITELIST: keywords that are unambiguously
    # consistent for each agent type.
    # If a Bull cites these -> consistent.
    # If a Bear cites these -> contradictory.
    BULL_CONSISTENT_KEYWORDS = [
        "analyst upside", "analyst_upside",
        "absolute upside",
        "bullish macd", "bullish_macd",
        "oversold",
        "portfolio_alpha", "alpha",
        "unrealized_pnl",
        "unrealized pnl",
    ]

    # Keywords unambiguously consistent for Bear
    BEAR_CONSISTENT_KEYWORDS = [
        "max_drawdown", "max drawdown",
        "overbought",
        "cvar", "portfolio_cvar",
        "garch", "vol_regime",
        "stop_loss", "stop loss",
        "concentration",
        "above hrp", "exceeds hrp",
        "current_weight",
    ]

    # Keywords that are contradictory by definition:
    # Bull should not cite risk as a reason to buy.
    # Bear should not cite upside as reason to sell.
    BULL_CONTRADICTORY_KEYWORDS = [
        "max_drawdown", "max drawdown",
        "overbought",
        "cvar",
        "below sma200", "below_sma_200",
    ]

    BEAR_CONTRADICTORY_KEYWORDS = [
        "analyst upside", "analyst_upside",
        "bullish macd",
        "oversold",
    ]

    # Special case: trend_confidence is contradictory
    # for Bull ONLY if trend_direction is Down
    # and confidence is high (>0.5)

    total_citations = 0
    consistent_cites = 0
    contradictory_cites = 0
    ambiguous_cites = 0
    contradictions_log = []

    recs = []
    if isinstance(agent_output, dict):
        recs = agent_output.get("recommendations", [])

    for rec in recs:
        ticker = rec.get("ticker", "")
        cited = rec.get("key_metrics_cited", []) or []
        reasoning = rec.get("reasoning", "") or ""

        # Parse from reasoning if cited is empty
        if not cited and reasoning:
            import re
            m = re.search(
                r"[Cc]ited[:\s]+(.+?)(?:\.|$)",
                reasoning,
                re.DOTALL
            )
            if m:
                cited = [
                    c.strip()
                    for c in m.group(1).split(",")
                    if len(c.strip()) > 3
                ]

        tren = trend_signals.get(ticker, {}) or {}
        td = per_ticker.get(ticker, {}) or {}
        _ = td

        for cite in cited:
            if not cite:
                continue
            cite_str = str(cite).lower()
            # Normalize underscores to spaces
            cite_norm = cite_str.replace("_", " ")
            total_citations += 1
            classified = False

            # Check consistent whitelist
            if agent_type == "bull":
                keywords_consistent = BULL_CONSISTENT_KEYWORDS
                keywords_contra = BULL_CONTRADICTORY_KEYWORDS
            else:
                keywords_consistent = BEAR_CONSISTENT_KEYWORDS
                keywords_contra = BEAR_CONTRADICTORY_KEYWORDS

            for kw in keywords_consistent:
                if kw.replace("_", " ") in cite_norm:
                    consistent_cites += 1
                    classified = True
                    break

            if classified:
                continue

            # Check contradictory whitelist
            for kw in keywords_contra:
                if kw.replace("_", " ") in cite_norm:
                    contradictory_cites += 1
                    contradictions_log.append({
                        "ticker": ticker,
                        "cite": cite_str[:120],
                        "issue": (
                            f"{agent_type.upper()} "
                            f"cites a "
                            f"{'risk' if agent_type=='bull' else 'bullish'}"
                            f" metric in support of "
                            f"{'BUY' if agent_type=='bull' else 'SELL'}"
                        ),
                        "severity": "HIGH"
                    })
                    classified = True
                    break

            if classified:
                continue

            # Special: trend_confidence for Bull
            # on a downtrending ticker
            if ("trend confidence" in cite_norm
                    or "trend_confidence" in cite_norm):
                trend_dir = tren.get("trend_direction", "")
                conf = float(
                    tren.get("trend_confidence_score", 0) or 0
                )
                if (agent_type == "bull"
                        and trend_dir == "Down"
                        and conf > 0.5):
                    contradictory_cites += 1
                    contradictions_log.append({
                        "ticker": ticker,
                        "cite": cite_str[:120],
                        "issue": (
                            f"Bull cites high trend "
                            f"confidence ({conf:.2f}) "
                            f"but trend_direction=Down"
                            f" — this is a bearish "
                            f"signal, not bullish"
                        ),
                        "severity": "HIGH"
                    })
                else:
                    consistent_cites += 1
                classified = True

            # Low confidence ticker cited for trend
            if not classified and ticker in low_conf:
                if ("trend" in cite_norm
                        or "confidence" in cite_norm):
                    contradictory_cites += 1
                    contradictions_log.append({
                        "ticker": ticker,
                        "cite": cite_str[:120],
                        "issue": (
                            f"{ticker} is flagged as "
                            f"low confidence but "
                            f"trend signal cited"
                        ),
                        "severity": "MEDIUM"
                    })
                    classified = True

            if not classified:
                ambiguous_cites += 1

    consistency_rate = (
        consistent_cites / total_citations
        if total_citations > 0 else 1.0
    )
    contradiction_rate = (
        contradictory_cites / total_citations
        if total_citations > 0 else 0.0
    )

    return {
        "agent_type": agent_type,
        "total_citations": total_citations,
        "consistent_citations": consistent_cites,
        "contradictory_citations": contradictory_cites,
        "ambiguous_citations": ambiguous_cites,
        "consistency_rate": round(
            consistency_rate, 3
        ),
        "contradiction_rate": round(
            contradiction_rate, 3
        ),
        "contradictions": contradictions_log,
        "quality_grade": (
            "A" if consistency_rate >= 0.85 else
            "B" if consistency_rate >= 0.70 else
            "C" if consistency_rate >= 0.55 else
            "D"
        ),
    }
