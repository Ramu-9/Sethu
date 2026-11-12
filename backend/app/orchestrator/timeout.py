"""
Caller silence timeout handling and simulated control-room notifications (Section 19).

Enforces Rule 5:
- Silence NEVER implies unconsciousness, death, injury, or abandonment.
- Silence NEVER cancels simulated assignments or marks an incident resolved.
- Silence NEVER increments the unknown-classification or provider-blocked counters.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeoutPolicyConfig:
    """
    Configurable timeout policy parameters (Section 19 & Gate 0.11).
    Must be explicitly provided when instantiating the timeout coordinator.
    """
    initial_silence_seconds: float
    max_check_ins: int
    wait_between_check_ins_seconds: float
    max_idle_exhaustion_seconds: float

    def __post_init__(self) -> None:
        if self.initial_silence_seconds <= 0:
            raise ValueError("initial_silence_seconds must be positive")
        if self.max_check_ins < 1:
            raise ValueError("max_check_ins must be at least 1")
        if self.wait_between_check_ins_seconds <= 0:
            raise ValueError("wait_between_check_ins_seconds must be positive")
        if self.max_idle_exhaustion_seconds <= self.initial_silence_seconds:
            raise ValueError("max_idle_exhaustion_seconds must exceed initial_silence_seconds")


@dataclass(frozen=True)
class TimeoutEvaluationResult:
    """Action determined for the current caller silence duration (Section 19)."""
    action: str  # 'no_action', 'play_check_in_prompt', 'flag_control_room_review', 'timeout_exhaustion_close_session'
    flag_control_room: bool
    session_exhausted: bool
    check_in_message_en: str | None


class CallerTimeoutCoordinator:
    """Evaluates caller silence against the configured TimeoutPolicyConfig (Section 19)."""

    def __init__(self, config: TimeoutPolicyConfig) -> None:
        self._config = config

    @property
    def config(self) -> TimeoutPolicyConfig:
        return self._config

    def evaluate_silence(
        self,
        *,
        idle_seconds: float,
        check_ins_sent: int,
    ) -> TimeoutEvaluationResult:
        """
        Determine timeout step without ever diagnosing unconsciousness or cancelling assignments.
        """
        if idle_seconds >= self._config.max_idle_exhaustion_seconds:
            return TimeoutEvaluationResult(
                action="timeout_exhaustion_close_session",
                flag_control_room=True,
                session_exhausted=True,
                check_in_message_en=(
                    "We have not heard a response on this simulated session. "
                    "Your simulated assignments remain preserved and flagged for control-room review."
                ),
            )

        if idle_seconds < self._config.initial_silence_seconds:
            return TimeoutEvaluationResult(
                action="no_action",
                flag_control_room=False,
                session_exhausted=False,
                check_in_message_en=None,
            )

        if check_ins_sent < self._config.max_check_ins:
            required_elapsed = (
                self._config.initial_silence_seconds
                + check_ins_sent * self._config.wait_between_check_ins_seconds
            )
            if idle_seconds >= required_elapsed:
                return TimeoutEvaluationResult(
                    action="play_check_in_prompt",
                    flag_control_room=False,
                    session_exhausted=False,
                    check_in_message_en=(
                        "I'm still here with you on the line. When you are ready, please let me know "
                        "if you are in a safe place."
                    ),
                )

        exhausted_check_ins_threshold = (
            self._config.initial_silence_seconds
            + self._config.max_check_ins * self._config.wait_between_check_ins_seconds
        )
        if idle_seconds >= exhausted_check_ins_threshold:
            return TimeoutEvaluationResult(
                action="flag_control_room_review",
                flag_control_room=True,
                session_exhausted=False,
                check_in_message_en=(
                    "No response received after check-in prompts. Flagging incident for simulated "
                    "control-room review while preserving all simulated state."
                ),
            )

        return TimeoutEvaluationResult(
            action="no_action",
            flag_control_room=False,
            session_exhausted=False,
            check_in_message_en=None,
        )
