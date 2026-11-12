"""
Google Gemini API reasoning adapter (Section 3.2, 9.2, 9.4, 23.2, and Gate 0.3).

Enforces:
- Strict separation between ProviderBlockedContentError ('provider_blocked_emergency_content'),
  ReasoningUnavailableError ('reasoning_unavailable'), and ReasoningSchemaValidationError.
- Bounded 1-retry policy (250ms default delay or Retry-After within deadline) for transient
  HTTP 408/429/5xx or connection errors only; never retries safety blocks, auth errors, or
  schema validation failures (Section 23.2).
- Never invents a fabricated schema fallback on validation failure.
- Never silently substitutes another model when GEMINI_MODEL_ID is configured.
"""

import asyncio
from collections.abc import Callable, Coroutine
import json
import time
from typing import Any
import urllib.error
import urllib.request
from uuid import UUID

from backend.app.adapters.base import (
    DialogueContext,
    EmergencyCategory,
    FactConflict,
    ProvenancedFact,
    ProviderBlockedContentError,
    ReasoningAdapter,
    ReasoningOutput,
    ReasoningSchemaValidationError,
    ReasoningUnavailableError,
    UtteranceRole,
)


DEFAULT_RETRY_DELAY_SECONDS: float = 0.250
BLOCKED_FINISH_REASONS = frozenset(
    {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "RECITATION", "SPII"}
)


def _normalize_identified_hazards(raw_hazards: Any) -> tuple[str, ...]:
    """Normalize extracted compound hazard identifiers into deduplicated lowercase snake_case tags."""
    if not isinstance(raw_hazards, list):
        return ()
    seen: list[str] = []
    for item in raw_hazards:
        tag = str(item or "").strip().lower().replace("-", "_").replace(" ", "_")
        if tag and tag not in seen:
            seen.append(tag)
    return tuple(seen)


def build_gemini_rest_transport(
    *,
    api_key: str,
    model_id: str = "gemini-3.8-flash",
) -> Callable[[DialogueContext, float], Coroutine[Any, Any, ReasoningOutput]]:
    """
    Build an async transport callable that invokes the Google Gemini API
    generateContent endpoint with structured JSON responseSchema enforcement
    and strict safety-block vs. outage vs. schema-validation error mapping.
    """

    def _sync_call(context: DialogueContext, timeout_sec: float) -> ReasoningOutput:
        t0 = time.monotonic()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"

        prior_lines = [
            f"- [{u.utterance_id}] (lang={u.language}, role={u.utterance_role.value if u.utterance_role else 'none'}): {u.text}"
            for u in context.prior_utterances
        ]
        prior_facts = [
            f"- {f.field_name}={f.value} (source_utterance_id={f.source_utterance_id}, evidence='{f.evidence_text}')"
            for f in context.validated_facts
        ]
        history_lines = [
            f"- [{t.role}] (lang={t.language or context.active_language}"
            + (f", utt={t.utterance_id}" if t.utterance_id else "")
            + f"): {t.text}"
            for t in context.conversation_history
        ]
        eligible_ids = list(context.eligible_instruction_ids) or ["disclosure_session_start"]

        pending_q = context.pending_question
        pending_q_summary = (
            f"question_id={pending_q.question_id}, target_field={pending_q.target_field}, "
            f"status={pending_q.status}, text='{pending_q.question_text}'"
            if pending_q is not None
            else (context.pending_followup_question or "None")
        )
        confirmed_field_keys = sorted({f.field_name for f in context.validated_facts if f.value not in (None, "")})

        system_instruction = (
            "You are SETU, a calm, reassuring, empathetic emergency intake voice assistant "
            "(an emergency coordination prototype for India; NOT 112). "
            "Communicate like a warm, steady human coordinator: briefly acknowledge what the caller said, "
            "reassure them to stay calm and take one step at a time, keep sentences concise (1 to 2 short natural sentences), "
            "and vary your acknowledgements naturally instead of repeating mechanical templates. "
            "Classify the incident strictly into one of the 9 categories: "
            "medical_emergency, fire, fire_with_injuries, road_accident, building_collapse, "
            "physical_attack, gas_leak, explosion, unknown. "
            "Classify utterance_role strictly into one of: new_description, "
            "answer_to_pending_question, acknowledgement, other. "
            "Pending Question Answer Rule (CRITICAL): "
            "If Pending Question State has a target_field and the caller replies cleanly "
            "(such as 'Yes', 'No', 'Two people', 'Near ABC building', 'Not breathing', or Telugu/Hindi/code-mixed replies), "
            "set utterance_role='answer_to_pending_question', set pending_question_answered=true, "
            "and extract that exact target_field into extracted_facts using the current utterance UUID and exact evidence_text. "
            "NEVER ask the same pending question again or ask for any field already in Confirmed Field Keys "
            "unless clarifying an explicit contradiction. "
            "Support English ('en'), Telugu ('te'), Hindi ('hi'), and Tanglish/Hinglish code-mixed speech naturally in Active Language. "
            "Conversational Memory Rules: "
            "Preserve all previously confirmed facts from Prior Validated Facts and Conversation History. "
            "If the caller refines location (e.g., adds street/landmark after an area name), extract both or preserve the broad area. "
            "Supported critical fields: emergency_type, broad_location_text, caller_location_text, caller_landmark_text, "
            "injuries_reported, affected_person_count, consciousness_status, hazard_details, caller_safe_spot_status. "
            "Also populate identified_hazards with ALL simultaneous distinct hazard tags detected across the situation "
            "(for example: vehicle_collision, fire, gas_cylinder, injuries, explosion, gas_leak, building_collapse, trapped_persons, assault). "
            "Ask AT MOST ONE short clarifying question in proposed_followup_question (with its field name in proposed_followup_target_field) "
            "ONLY when genuinely missing or ambiguous; otherwise leave proposed_followup_question empty. "
            "Every extracted fact MUST reference a valid UUID from the provided utterances "
            "and quote an exact substring evidence_text from that utterance. "
            "If a newly extracted fact contradicts a Previously Validated Fact for the same field_name, "
            "include a conflict entry in conflicts with both utterance UUIDs and values. "
            "Never invent unit IDs, vehicle assignments, ETAs, or dispatch confirmation (backend handles dispatch after explicit location confirmation). "
            "Never generate unapproved medical or hazardous instructions; select the most relevant selected_instruction_id "
            f"only from: {', '.join(eligible_ids)}. "
            "If the emergency is already established and location confirmation is pending or units are simulated EN_ROUTE, "
            "provide calm situational reassurance and safety guidance aligned with the selected approved instruction. "
            "Never claim real 112 dispatch."
        )

        hazards_summary = ", ".join(context.identified_hazards) if context.identified_hazards else "None"
        prompt_text = (
            f"Session ID: {context.session_id}\n"
            f"Incident ID: {context.incident_id}\n"
            f"Active Language: {context.active_language}\n"
            f"Backend Status: {json.dumps(dict(context.backend_status_summary), default=str)}\n"
            f"Confirmed Field Keys (DO NOT RE-ASK): {json.dumps(confirmed_field_keys)}\n"
            f"Known Simultaneous Hazards So Far: {hazards_summary}\n"
            f"Pending Question State: {pending_q_summary}\n"
            f"Session Conversation History:\n{chr(10).join(history_lines) if history_lines else 'None'}\n"
            f"Prior Utterances:\n{chr(10).join(prior_lines) if prior_lines else 'None'}\n"
            f"Previously Validated Facts:\n"
            f"{chr(10).join(prior_facts) if prior_facts else 'None'}\n"
            f"Current Finalized Utterance:\n"
            f"- [{context.current_utterance.utterance_id}] (lang={context.current_utterance.language}): {context.current_utterance.text}\n"
        )

        gen_config: dict[str, Any] = {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "utterance_role": {
                        "type": "STRING",
                        "enum": [r.value for r in UtteranceRole],
                    },
                    "category": {
                        "type": "STRING",
                        "enum": [c.value for c in EmergencyCategory],
                    },
                    "category_confidence": {"type": "NUMBER"},
                    "extracted_facts": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "field_name": {"type": "STRING"},
                                "value": {"type": "STRING"},
                                "source_utterance_id": {"type": "STRING"},
                                "evidence_text": {"type": "STRING"},
                            },
                            "required": [
                                "field_name",
                                "value",
                                "source_utterance_id",
                                "evidence_text",
                            ],
                        },
                    },
                    "conflicts": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "field_name": {"type": "STRING"},
                                "prior_value": {"type": "STRING"},
                                "prior_utterance_id": {"type": "STRING"},
                                "new_value": {"type": "STRING"},
                                "new_utterance_id": {"type": "STRING"},
                                "description": {"type": "STRING"},
                            },
                            "required": [
                                "field_name",
                                "prior_value",
                                "prior_utterance_id",
                                "new_value",
                                "new_utterance_id",
                            ],
                        },
                    },
                    "missing_information_fields": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "identified_hazards": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "pending_question_answered": {"type": "BOOLEAN"},
                    "proposed_followup_target_field": {"type": "STRING"},
                    "proposed_followup_question": {"type": "STRING"},
                    "selected_instruction_id": {"type": "STRING"},
                    "caller_response_text": {"type": "STRING"},
                },
                "required": [
                    "utterance_role",
                    "category",
                    "category_confidence",
                    "extracted_facts",
                    "selected_instruction_id",
                    "caller_response_text",
                ],
            },
        }
        if model_id.startswith("gemini-3.8"):
            gen_config["thinkingConfig"] = {"thinkingLevel": "LOW"}

        request_body = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
            "generationConfig": gen_config,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
                "User-Agent": "aistudio-build",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=max(0.5, timeout_sec)) as resp:
                raw_bytes = resp.read()
        except urllib.error.HTTPError as exc:
            retry_after_hdr = exc.headers.get("Retry-After") if exc.headers else None
            retry_after_sec: float | None = None
            if retry_after_hdr:
                try:
                    retry_after_sec = float(retry_after_hdr)
                except ValueError:
                    retry_after_sec = None
            err_detail = ""
            try:
                raw_err = exc.read().decode("utf-8", errors="ignore")
                parsed_err = json.loads(raw_err)
                msg = (parsed_err.get("error") or {}).get("message")
                if msg:
                    err_detail = f" ({str(msg).splitlines()[0][:180]})"
            except Exception:
                pass
            is_retryable = exc.code in {408, 429, 500, 502, 503, 504}
            raise ReasoningUnavailableError(
                f"Gemini HTTP {exc.code}: {exc.reason}{err_detail}",
                session_id=context.session_id,
                utterance_id=context.current_utterance.utterance_id,
                is_retryable=is_retryable,
                http_status=exc.code,
                retry_after_seconds=retry_after_sec,
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ReasoningUnavailableError(
                f"Gemini network/timeout failure: {exc}",
                session_id=context.session_id,
                utterance_id=context.current_utterance.utterance_id,
                is_retryable=True,
                http_status=408,
            ) from exc

        try:
            envelope = json.loads(raw_bytes.decode("utf-8"))
        except Exception as exc:
            raise ReasoningSchemaValidationError(
                f"Invalid JSON envelope from Gemini: {exc}",
                session_id=context.session_id,
                utterance_id=context.current_utterance.utterance_id,
            ) from exc

        prompt_feedback = envelope.get("promptFeedback") or {}
        if prompt_feedback.get("blockReason"):
            raise ProviderBlockedContentError(
                f"Gemini blocked prompt with blockReason={prompt_feedback['blockReason']}",
                session_id=context.session_id,
                utterance_id=context.current_utterance.utterance_id,
                provider_finish_reason=str(prompt_feedback["blockReason"]),
            )

        candidates = envelope.get("candidates") or []
        if not candidates:
            raise ProviderBlockedContentError(
                "Gemini returned empty candidates array (treated as safety block per Section 9.4).",
                session_id=context.session_id,
                utterance_id=context.current_utterance.utterance_id,
                provider_finish_reason="EMPTY_CANDIDATES",
            )

        first_cand = candidates[0]
        finish_reason = str(first_cand.get("finishReason", "STOP"))
        if finish_reason in BLOCKED_FINISH_REASONS:
            raise ProviderBlockedContentError(
                f"Gemini blocked candidate with finishReason={finish_reason}",
                session_id=context.session_id,
                utterance_id=context.current_utterance.utterance_id,
                provider_finish_reason=finish_reason,
            )

        parts = (first_cand.get("content") or {}).get("parts") or []
        raw_text = "".join(str(p.get("text", "")) for p in parts if not p.get("thought")).strip()
        if not raw_text:
            raw_text = "".join(str(p.get("text", "")) for p in parts).strip()
        if not raw_text:
            raise ReasoningSchemaValidationError(
                "Gemini returned empty text part.",
                session_id=context.session_id,
                utterance_id=context.current_utterance.utterance_id,
            )

        try:
            parsed = json.loads(raw_text)
            role = UtteranceRole(str(parsed["utterance_role"]))
            cat = EmergencyCategory(str(parsed["category"]))
            conf = float(parsed["category_confidence"])
            facts: list[ProvenancedFact] = []
            for rf in parsed.get("extracted_facts") or []:
                fname = str(rf["field_name"])
                raw_val: bool | int | str | None = rf["value"]
                if isinstance(raw_val, str):
                    lowered_val = raw_val.strip().lower()
                    if lowered_val == "true":
                        raw_val = True
                    elif lowered_val == "false":
                        raw_val = False
                    elif fname == "affected_person_count" and lowered_val.isdigit():
                        raw_val = int(lowered_val)
                facts.append(
                    ProvenancedFact(
                        field_name=fname,
                        value=raw_val,
                        source_utterance_id=UUID(str(rf["source_utterance_id"])),
                        evidence_text=str(rf["evidence_text"]),
                    )
                )
            conflicts_list: list[FactConflict] = []
            for rc in parsed.get("conflicts") or []:
                conflicts_list.append(
                    FactConflict(
                        field_name=str(rc["field_name"]),
                        prior_value=rc["prior_value"],
                        prior_utterance_id=UUID(str(rc["prior_utterance_id"])),
                        new_value=rc["new_value"],
                        new_utterance_id=UUID(str(rc["new_utterance_id"])),
                        description=str(rc.get("description") or ""),
                    )
                )
            missing_fields = tuple(str(m) for m in (parsed.get("missing_information_fields") or []))
            inst_id = str(parsed.get("selected_instruction_id") or eligible_ids[0])
            if inst_id not in eligible_ids:
                inst_id = eligible_ids[0]

            usage = envelope.get("usageMetadata") or {}
            latency_ms = round((time.monotonic() - t0) * 1000.0, 2)
            return ReasoningOutput(
                utterance_role=role,
                utterance_role_source_id=context.current_utterance.utterance_id,
                category=cat,
                category_confidence=conf,
                supporting_utterance_ids=(context.current_utterance.utterance_id,),
                extracted_facts=tuple(facts),
                conflicts=tuple(conflicts_list),
                missing_information_fields=missing_fields,
                proposed_followup_question=parsed.get("proposed_followup_question") or None,
                selected_instruction_id=inst_id,
                caller_response_text=str(parsed["caller_response_text"]),
                prompt_tokens=int(usage.get("promptTokenCount", 0)),
                completion_tokens=int(usage.get("candidatesTokenCount", 0)),
                latency_ms=latency_ms,
                pending_question_answered=(
                    bool(parsed["pending_question_answered"])
                    if parsed.get("pending_question_answered") is not None
                    else None
                ),
                proposed_followup_target_field=(
                    str(parsed["proposed_followup_target_field"]).strip() or None
                    if parsed.get("proposed_followup_target_field")
                    else None
                ),
                identified_hazards=_normalize_identified_hazards(parsed.get("identified_hazards")),
            )
        except Exception as exc:
            raise ReasoningSchemaValidationError(
                f"Gemini structured output failed schema validation: {exc}",
                session_id=context.session_id,
                utterance_id=context.current_utterance.utterance_id,
            ) from exc

    async def _async_transport(context: DialogueContext, timeout_sec: float) -> ReasoningOutput:
        return await asyncio.to_thread(_sync_call, context, timeout_sec)

    return _async_transport


class GeminiReasoningAdapter(ReasoningAdapter):
    """
    Provider-agnostic Google Gemini reasoning adapter with bounded 1-retry policy
    and explicit safety-block vs. outage error mapping.
    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        transport_callable: Callable[[DialogueContext, float], Coroutine[Any, Any, ReasoningOutput]] | None = None,
    ) -> None:
        self._model_id = model_id
        self._transport = transport_callable

    @property
    def model_id(self) -> str | None:
        return self._model_id

    @property
    def is_configured(self) -> bool:
        return bool(self._model_id and self._transport is not None)

    async def analyze_utterance(
        self,
        context: DialogueContext,
        *,
        timeout_seconds: float,
    ) -> ReasoningOutput:
        """
        Execute structured classification and fact extraction with full dialogue context.
        """
        if self._transport is None:
            raise ReasoningUnavailableError(
                "Gate 0.3 unresolved: Gemini model identifier and transport client not configured.",
                session_id=context.session_id,
                utterance_id=context.current_utterance.utterance_id,
                is_retryable=False,
            )

        deadline = time.monotonic() + timeout_seconds
        attempts = 0
        max_attempts = 2  # Initial attempt + at most ONE retry (Section 23.2)

        while attempts < max_attempts:
            attempts += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise ReasoningUnavailableError(
                    "Reasoning operation exceeded total deadline before attempt.",
                    session_id=context.session_id,
                    utterance_id=context.current_utterance.utterance_id,
                    is_retryable=False,
                )

            try:
                output = await asyncio.wait_for(
                    self._transport(context, remaining),
                    timeout=remaining,
                )
                output.validate_against_context(context)
                return output
            except (ProviderBlockedContentError, ReasoningSchemaValidationError):
                # Never retry safety-filter blocks or schema validation failures (Section 23.2)
                raise
            except asyncio.TimeoutError as exc:
                err = ReasoningUnavailableError(
                    "Reasoning request timed out.",
                    session_id=context.session_id,
                    utterance_id=context.current_utterance.utterance_id,
                    is_retryable=True,
                    http_status=408,
                )
                if attempts >= max_attempts:
                    raise err from exc
                retry_delay = DEFAULT_RETRY_DELAY_SECONDS
            except ReasoningUnavailableError as exc:
                if not exc.is_retryable or attempts >= max_attempts:
                    raise
                retry_delay = (
                    exc.retry_after_seconds
                    if exc.retry_after_seconds is not None and exc.retry_after_seconds > 0
                    else DEFAULT_RETRY_DELAY_SECONDS
                )

            # Honour Retry-After only if it fits within the remaining operation deadline (Section 23.2)
            remaining_after_failure = deadline - time.monotonic()
            if retry_delay >= remaining_after_failure:
                raise ReasoningUnavailableError(
                    "Retry delay exceeds remaining operation deadline.",
                    session_id=context.session_id,
                    utterance_id=context.current_utterance.utterance_id,
                    is_retryable=False,
                )
            await asyncio.sleep(retry_delay)

        raise ReasoningUnavailableError(
            "Exhausted bounded retry policy.",
            session_id=context.session_id,
            utterance_id=context.current_utterance.utterance_id,
            is_retryable=False,
        )
