from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class EventMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    mean_trigger_delay: float | None
    duplicate_triggers_per_event: float
    early_triggers: int
    early_trigger_rate: float
    triggers_per_100_frames: float


def average_precision(labels: Iterable[float], scores: Iterable[float]) -> float:
    y_true = np.asarray(list(labels), dtype=np.float64)
    y_score = np.asarray(list(scores), dtype=np.float64)
    if y_true.size == 0 or np.sum(y_true > 0) == 0:
        return 0.0
    order = np.argsort(-y_score)
    y_true = (y_true[order] > 0).astype(np.float64)
    tp = np.cumsum(y_true)
    fp = np.cumsum(1.0 - y_true)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall_delta = y_true / max(np.sum(y_true), 1.0)
    return float(np.sum(precision * recall_delta))


def detect_peaks(scores: list[float], threshold: float, *, cooldown: int = 0, local_max: bool = True) -> list[int]:
    peaks: list[int] = []
    last_peak = -10**9
    for index, score in enumerate(scores):
        if score < threshold:
            continue
        if local_max:
            left = scores[index - 1] if index > 0 else -float("inf")
            right = scores[index + 1] if index + 1 < len(scores) else -float("inf")
            if score < left or score < right:
                continue
        if index - last_peak <= cooldown:
            if peaks and score > scores[peaks[-1]]:
                peaks[-1] = index
                last_peak = index
            continue
        peaks.append(index)
        last_peak = index
    return peaks


def match_events(
    predicted: list[int],
    truth: list[int],
    tolerance: int | None = None,
    *,
    min_delay: int | None = None,
    max_delay: int | None = None,
    early_tolerance: int | None = None,
) -> EventMetrics:
    if tolerance is None:
        tolerance = 0
    if min_delay is None:
        min_delay = -int(tolerance)
    if max_delay is None:
        max_delay = int(tolerance)
    if min_delay > max_delay:
        raise ValueError("min_delay must be <= max_delay")
    if early_tolerance is None:
        early_tolerance = max(abs(min_delay), abs(max_delay), int(tolerance))

    matched_truth: set[int] = set()
    delays: list[int] = []
    true_positives = 0
    false_positives = 0
    duplicate_triggers = 0
    early_triggers = 0

    for pred in predicted:
        candidates = [
            (abs(pred - event), event)
            for event in truth
            if min_delay <= pred - event <= max_delay and event not in matched_truth
        ]
        if candidates:
            _, event = min(candidates)
            matched_truth.add(event)
            true_positives += 1
            delays.append(pred - event)
        elif any(min_delay <= pred - event <= max_delay for event in truth):
            duplicate_triggers += 1
            false_positives += 1
        else:
            false_positives += 1
            if _is_early_trigger(pred, truth, min_delay=min_delay, early_tolerance=early_tolerance):
                early_triggers += 1

    false_negatives = len(truth) - true_positives
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    mean_delay = float(np.mean(delays)) if delays else None
    duplicate_rate = duplicate_triggers / max(len(truth), 1)
    early_rate = early_triggers / max(len(truth), 1)
    return EventMetrics(
        threshold=0.0,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        mean_trigger_delay=mean_delay,
        duplicate_triggers_per_event=duplicate_rate,
        early_triggers=early_triggers,
        early_trigger_rate=early_rate,
        triggers_per_100_frames=0.0,
    )


def _is_early_trigger(pred: int, truth: list[int], *, min_delay: int, early_tolerance: int) -> bool:
    for event in truth:
        first_valid = event + min_delay
        if pred < first_valid and first_valid - pred <= early_tolerance:
            return True
    return False


def evaluate_event_grid(
    grouped_scores: dict[str, list[tuple[int, float]]],
    grouped_events: dict[str, list[int]],
    thresholds: Iterable[float],
    *,
    tolerance: int,
    cooldown: int,
    min_delay: int | None = None,
    max_delay: int | None = None,
    early_tolerance: int | None = None,
) -> list[EventMetrics]:
    results: list[EventMetrics] = []
    for threshold in thresholds:
        all_predicted: list[int] = []
        all_truth: list[int] = []
        total_scored_frames = 0
        offsets = 0
        for trajectory_id, frame_scores in grouped_scores.items():
            if not frame_scores:
                continue
            frame_scores = sorted(frame_scores)
            frames = [frame for frame, _ in frame_scores]
            scores = [score for _, score in frame_scores]
            peaks = detect_peaks(scores, float(threshold), cooldown=cooldown, local_max=True)
            all_predicted.extend(offsets + frames[index] for index in peaks)
            all_truth.extend(offsets + event for event in grouped_events.get(trajectory_id, []))
            total_scored_frames += len(frame_scores)
            offsets += max(frames) + max(tolerance, abs(min_delay or 0), abs(max_delay or 0)) + cooldown + 100
        metrics = match_events(
            all_predicted,
            all_truth,
            tolerance,
            min_delay=min_delay,
            max_delay=max_delay,
            early_tolerance=early_tolerance,
        )
        results.append(
            EventMetrics(
                threshold=float(threshold),
                precision=metrics.precision,
                recall=metrics.recall,
                f1=metrics.f1,
                true_positives=metrics.true_positives,
                false_positives=metrics.false_positives,
                false_negatives=metrics.false_negatives,
                mean_trigger_delay=metrics.mean_trigger_delay,
                duplicate_triggers_per_event=metrics.duplicate_triggers_per_event,
                early_triggers=metrics.early_triggers,
                early_trigger_rate=metrics.early_trigger_rate,
                triggers_per_100_frames=len(all_predicted) / max(total_scored_frames, 1) * 100.0,
            )
        )
    return results
