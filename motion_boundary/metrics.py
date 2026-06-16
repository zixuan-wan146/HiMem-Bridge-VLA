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


def match_events(predicted: list[int], truth: list[int], tolerance: int) -> EventMetrics:
    matched_truth: set[int] = set()
    delays: list[int] = []
    true_positives = 0
    false_positives = 0
    duplicate_triggers = 0

    for pred in predicted:
        candidates = [
            (abs(pred - event), event)
            for event in truth
            if abs(pred - event) <= tolerance and event not in matched_truth
        ]
        if candidates:
            _, event = min(candidates)
            matched_truth.add(event)
            true_positives += 1
            delays.append(pred - event)
        elif any(abs(pred - event) <= tolerance for event in truth):
            duplicate_triggers += 1
            false_positives += 1
        else:
            false_positives += 1

    false_negatives = len(truth) - true_positives
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    mean_delay = float(np.mean(delays)) if delays else None
    duplicate_rate = duplicate_triggers / max(len(truth), 1)
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
    )


def evaluate_event_grid(
    grouped_scores: dict[str, list[tuple[int, float]]],
    grouped_events: dict[str, list[int]],
    thresholds: Iterable[float],
    *,
    tolerance: int,
    cooldown: int,
) -> list[EventMetrics]:
    results: list[EventMetrics] = []
    for threshold in thresholds:
        all_predicted: list[int] = []
        all_truth: list[int] = []
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
            offsets += (max(frames) + tolerance + cooldown + 100)
        metrics = match_events(all_predicted, all_truth, tolerance)
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
            )
        )
    return results
