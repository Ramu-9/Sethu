"""
Backend policy validation, caller-facing text safety validation, and
possible false/prank call flag review workflow (Sections 3.2, 5, 11, and 23).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import re
from uuid import UUID

from backend.app.adapters.base import (
    DialogueContext,
    ReasoningOutput,
    SupportedLanguage,
)
from backend.app.orchestrator.instructions import InstructionLibrary


class PrankFlagReviewStatus(str, Enum):
    """Possible review statuses for possible_false_or_prank_call (Section 11)."""
    NOT_REVIEWED = "not_reviewed"
    FLAGGED_FOR_REVIEW = "flagged_for_review"
    CONFIRMED_FALSE_OR_PRANK = "confirmed_false_or_prank"
    NOT_FALSE_OR_PRANK = "not_false_or_prank"


_PROHIBITED_PRANK_REASONS = frozenset(
    {
        "silence",
        "caller_silence",
        "unclear_speech",
        "language_barrier",
        "language_difficulty",
        "provider_blocked",
        "provider_blocked_emergency_content",
        "incomplete_information",
        "inconsistent_information",
        "llm_confirmed_prank",
    }
)


@dataclass(frozen=True)
class PossiblePrankCallFlag:
    """Auditable review indicator for a possible false or prank call (Section 11)."""
    possible_false_or_prank_call: bool
    flag_reason: str
    flag_source: str
    flag_timestamp: datetime
    review_status: PrankFlagReviewStatus
    reviewer_id: str | None
    supporting_utterance_ids: tuple[UUID, ...]


def create_possible_prank_flag(
    *,
    flag_reason: str,
    flag_source: str,
    supporting_utterance_ids: Sequence[UUID],
    now: datetime,
) -> PossiblePrankCallFlag:
    """
    Create a review flag for a possible false/prank call (Section 11).

    Rules:
    - Silence, unclear speech, language barriers, or provider blocks are NEVER valid evidence.
    - The reasoning model cannot independently declare a confirmed prank caller.
    - Initial status is always FLAGGED_FOR_REVIEW, never CONFIRMED_FALSE_OR_PRANK.
    """
    normalized_reason = flag_reason.strip().lower()
    if normalized_reason in _PROHIBITED_PRANK_REASONS:
        raise ValueError(
            f"Reason '{flag_reason}' is prohibited by Section 11 as evidence of a prank call."
        )
    if not supporting_utterance_ids:
        raise ValueError("Supporting utterance or event references are required for a prank flag.")

    return PossiblePrankCallFlag(
        possible_false_or_prank_call=True,
        flag_reason=flag_reason,
        flag_source=flag_source,
        flag_timestamp=now,
        review_status=PrankFlagReviewStatus.FLAGGED_FOR_REVIEW,
        reviewer_id=None,
        supporting_utterance_ids=tuple(supporting_utterance_ids),
    )


def update_prank_flag_review(
    flag: PossiblePrankCallFlag,
    *,
    new_status: PrankFlagReviewStatus,
    reviewer_id: str,
    reviewed_at: datetime,
) -> PossiblePrankCallFlag:
    """Update review status by an authorised human/simulated reviewer (Section 11)."""
    if not reviewer_id or not reviewer_id.strip():
        raise ValueError("reviewer_id is required to update prank flag review status")
    return PossiblePrankCallFlag(
        possible_false_or_prank_call=(new_status != PrankFlagReviewStatus.NOT_FALSE_OR_PRANK),
        flag_reason=flag.flag_reason,
        flag_source=flag.flag_source,
        flag_timestamp=reviewed_at,
        review_status=new_status,
        reviewer_id=reviewer_id.strip(),
        supporting_utterance_ids=flag.supporting_utterance_ids,
    )


# Patterns that falsely claim real-world dispatch or 112 connection (Section 2.1 & 3.2)
_FORBIDDEN_REAL_DISPATCH_PATTERNS = (
    re.compile(r"\b(?:called|contacted|connected\s+to|transferred\s+to)\s+112\b", re.I),
    re.compile(r"\breal\s+(?:ambulance|police|fire\s+engine|help|dispatch)\b", re.I),
    re.compile(r"\b(?:ambulance|police|fire\s+truck)\s+is\s+on\s+(?:its|the)\s+way\b", re.I),
    re.compile(r"\bdispatched\s+a\s+real\b", re.I),
)


class PolicyValidator:
    """
    Validates reasoning outputs, provenance, instruction eligibility, and caller-facing text (Section 23).
    """

    def __init__(self, instruction_library: InstructionLibrary) -> None:
        self._library = instruction_library

    def validate_reasoning_output(
        self,
        output: ReasoningOutput,
        context: DialogueContext,
    ) -> ReasoningOutput:
        """
        Enforce schema, provenance, and instruction-ID validation (Section 23).
        """
        output.validate_against_context(context)
        return output

    def sanitize_caller_response_text(
        self,
        text: str,
        *,
        language: SupportedLanguage,
        has_simulated_assignment: bool,
    ) -> tuple[str, bool]:
        """
        Section 3.2: Caller-facing dynamic text must never state or imply that real-world
        help has been requested, dispatched, or is arriving. Falls back to an approved
        cached phrase if validation fails.
        """
        cleaned = text.strip()
        if not cleaned:
            reassurance = self._library.get_outage_reassurance_sequence(language)[0]
            return reassurance.approved_text, True

        for pattern in _FORBIDDEN_REAL_DISPATCH_PATTERNS:
            if pattern.search(cleaned):
                if has_simulated_assignment:
                    return (
                        "In this demo, a simulated resource assignment has been created. "
                        "This prototype is not connected to 112 or real emergency services.",
                        True,
                    )
                reassurance = self._library.get_outage_reassurance_sequence(language)[0]
                return reassurance.approved_text, True

        return cleaned, False
