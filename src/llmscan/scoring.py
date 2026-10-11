"""Turning per-detector findings into one number per file, directory and repository."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import FileReport, Finding

# Files at or above this combined score are counted as "flagged"...
FLAG_THRESHOLD = 0.65
# ...but only when the detectors are at least this sure; tiny files give noisy scores.
MIN_CONFIDENCE = 0.4


@dataclass
class Aggregate:
    score: float | None
    files: int
    scored_files: int
    loc: int
    flagged_files: int
    flagged_loc: int

    @property
    def flagged_share(self) -> float:
        return self.flagged_loc / self.loc if self.loc else 0.0


def combine(findings: dict[str, Finding], weights: dict[str, float]) -> float | None:
    """Confidence-weighted mean of the detectors that have an opinion."""
    total = weight_sum = 0.0
    for name, finding in findings.items():
        if finding.score is None:
            continue
        weight = weights.get(name, 1.0) * finding.confidence
        total += finding.score * weight
        weight_sum += weight
    return total / weight_sum if weight_sum > 0 else None


def confidence(findings: dict[str, Finding], weights: dict[str, float]) -> float:
    """How much to trust combine(): the detectors' confidences, averaged by detector weight."""
    total = weight_sum = 0.0
    for name, finding in findings.items():
        if finding.score is None:
            continue
        weight = weights.get(name, 1.0)
        total += finding.confidence * weight
        weight_sum += weight
    return total / weight_sum if weight_sum > 0 else 0.0


def shrunk(score: float | None, certainty: float) -> float:
    """Score pulled towards the neutral 0.5 by uncertainty; used for ranking only."""
    return -1.0 if score is None else 0.5 + (score - 0.5) * certainty


def aggregate(reports: Iterable[FileReport], weights: dict[str, float]) -> Aggregate:
    """Mean over files weighted by lines and confidence: a big, clearly scored file counts for more than a noisy stub."""
    files = scored = loc = flagged_files = flagged_loc = 0
    weighted = scored_loc = 0.0
    for report in reports:
        files += 1
        loc += report.loc
        score = combine(report.findings, weights)
        if score is None:
            continue
        certainty = confidence(report.findings, weights)
        scored += 1
        weighted += score * report.loc * certainty
        scored_loc += report.loc * certainty
        if score >= FLAG_THRESHOLD and certainty >= MIN_CONFIDENCE:
            flagged_files += 1
            flagged_loc += report.loc
    return Aggregate(weighted / scored_loc if scored_loc else None, files, scored, loc, flagged_files, flagged_loc)
