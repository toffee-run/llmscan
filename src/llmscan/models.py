"""Plain data types shared by the scanner, the detectors and the UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceFile:
    """A file from the scanned repository. The text is read once and never written back."""

    path: Path
    rel: str
    text: str
    language: str
    sha: str

    @property
    def lines(self) -> list[str]:
        return self.text.splitlines()

    @property
    def loc(self) -> int:
        return sum(1 for line in self.text.splitlines() if line.strip())


@dataclass
class Evidence:
    """One human-readable reason behind a score, optionally tied to a line."""

    message: str
    line: int | None = None
    end_line: int | None = None
    score: float | None = None


@dataclass
class Finding:
    """What one detector thinks about one file.

    score       0..1, how LLM-like the file looks; None means "no opinion".
    confidence  0..1, how much this finding should weigh against other detectors.
    """

    score: float | None
    confidence: float = 1.0
    summary: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Finding":
        evidence = [Evidence(**e) for e in data.get("evidence", [])]
        return cls(
            score=data.get("score"),
            confidence=data.get("confidence", 1.0),
            summary=data.get("summary", ""),
            evidence=evidence,
            metrics=data.get("metrics", {}),
        )


@dataclass
class FileReport:
    """Everything known about one file: findings keyed by detector name."""

    rel: str
    language: str
    loc: int
    findings: dict[str, Finding] = field(default_factory=dict)
