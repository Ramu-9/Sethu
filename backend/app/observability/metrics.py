"""
Observability and TTFAB latency / provider usage instrumentation (Section 24).

Reports cached instruction playback separately from live Timbre TTFAB,
tracks sample counts for every metric, and exposes p50/p95 only when >= 30
valid observations exist (Section 26.4).
"""

from dataclasses import dataclass
import math
import threading


MIN_OBSERVATIONS_FOR_PERCENTILES: int = 30


@dataclass(frozen=True)
class MetricSummary:
    """Statistical summary of a measured latency metric (Section 24 & 26.4)."""
    metric_name: str
    sample_count: int
    last_value_ms: float | None
    min_ms: float | None
    max_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    percentiles_status: str


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        raise ValueError("Cannot compute percentile on empty list")
    idx = (len(sorted_vals) - 1) * q
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return round(sorted_vals[lower], 2)
    weight = idx - lower
    return round(sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight, 2)


class MetricsRegistry:
    """
    Real latency and usage instrumentation per Section 24.
    Never invents performance numbers and keeps cached playback separate from live TTS TTFAB.
    """

    SUPPORTED_METRICS = (
        "vad_finalize_ms",
        "prisma_partial_transcript_ms",
        "prisma_final_transcript_ms",
        "reasoning_processing_ms",
        "orchestrator_ms",
        "db_lookup_ms",
        "hospital_ranking_ms",
        "resource_lookup_ms",
        "assignment_creation_ms",
        "timbre_live_ttfab_ms",
        "timbre_total_ms",
        "cached_audio_lookup_ms",
        "cached_audio_playback_start_ms",
        "e2e_conversational_response_ms",
        "barge_in_local_stop_ms",
        "server_cancellation_ms",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: dict[str, list[float]] = {m: [] for m in self.SUPPORTED_METRICS}
        self._prompt_tokens_total = 0
        self._completion_tokens_total = 0
        self._rate_limit_count = 0
        self._provider_block_count = 0
        self._barge_in_count = 0
        self._false_barge_in_count = 0
        self._tool_timeout_count = 0

    def record_latency(self, metric_name: str, duration_ms: float) -> None:
        """Record a measured latency sample in milliseconds."""
        if duration_ms < 0.0:
            raise ValueError("duration_ms must be non-negative")
        with self._lock:
            bucket = self._samples.setdefault(metric_name, [])
            bucket.append(round(duration_ms, 2))

    def record_token_usage(
        self,
        *,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        with self._lock:
            if prompt_tokens is not None and prompt_tokens > 0:
                self._prompt_tokens_total += prompt_tokens
            if completion_tokens is not None and completion_tokens > 0:
                self._completion_tokens_total += completion_tokens

    def increment_provider_block(self) -> None:
        with self._lock:
            self._provider_block_count += 1

    def increment_rate_limit(self) -> None:
        with self._lock:
            self._rate_limit_count += 1

    def increment_barge_in(self, *, is_false_trigger: bool = False) -> None:
        with self._lock:
            self._barge_in_count += 1
            if is_false_trigger:
                self._false_barge_in_count += 1

    def increment_tool_timeout(self) -> None:
        with self._lock:
            self._tool_timeout_count += 1

    def get_metric_summary(self, metric_name: str) -> MetricSummary:
        """
        Return sample count, last value, and p50/p95 only when sample_count >= 30 (Section 26.4).
        """
        with self._lock:
            vals = list(self._samples.get(metric_name, ()))
        n = len(vals)
        if n == 0:
            return MetricSummary(
                metric_name=metric_name,
                sample_count=0,
                last_value_ms=None,
                min_ms=None,
                max_ms=None,
                p50_ms=None,
                p95_ms=None,
                percentiles_status="no_samples_recorded",
            )
        sorted_vals = sorted(vals)
        if n < MIN_OBSERVATIONS_FOR_PERCENTILES:
            return MetricSummary(
                metric_name=metric_name,
                sample_count=n,
                last_value_ms=vals[-1],
                min_ms=sorted_vals[0],
                max_ms=sorted_vals[-1],
                p50_ms=None,
                p95_ms=None,
                percentiles_status=f"insufficiently_sampled ({n}/{MIN_OBSERVATIONS_FOR_PERCENTILES})",
            )
        return MetricSummary(
            metric_name=metric_name,
            sample_count=n,
            last_value_ms=vals[-1],
            min_ms=sorted_vals[0],
            max_ms=sorted_vals[-1],
            p50_ms=_percentile(sorted_vals, 0.50),
            p95_ms=_percentile(sorted_vals, 0.95),
            percentiles_status="sufficiently_sampled",
        )

    def snapshot(self, *, cache_hits: int, cache_misses: int) -> dict[str, object]:
        """Export complete observability snapshot for engineering diagnostics (Section 24 & 25.6)."""
        total_cache = cache_hits + cache_misses
        hit_rate = round(cache_hits / total_cache, 4) if total_cache > 0 else None
        summaries = {
            m: self.get_metric_summary(m).__dict__
            for m in self.SUPPORTED_METRICS
        }
        with self._lock:
            return {
                "metrics": summaries,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "cache_hit_rate": hit_rate,
                "prompt_tokens_total": self._prompt_tokens_total,
                "completion_tokens_total": self._completion_tokens_total,
                "rate_limit_count": self._rate_limit_count,
                "provider_block_count": self._provider_block_count,
                "barge_in_count": self._barge_in_count,
                "false_barge_in_count": self._false_barge_in_count,
                "tool_timeout_count": self._tool_timeout_count,
            }
