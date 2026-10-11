"""Exporting results. Reports are refused if they would land inside the scanned repository."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from .engine import Scanner
from .repo import is_inside
from .storage import STATE_DIR


def build(scanner: Scanner, skip: Callable[[str], bool] = lambda rel: False) -> dict:
    total = scanner.summary(skip)
    files = []
    for rel, report in scanner.reports.items():
        files.append({
            "path": rel,
            "language": report.language,
            "loc": report.loc,
            "excluded": skip(rel),
            "score": scanner.score(rel),
            "confidence": scanner.confidence(rel),
            "findings": {name: finding.to_dict() for name, finding in report.findings.items()},
        })
    files.sort(key=lambda f: -scanner.rank(f["path"]))
    return {
        "root": str(scanner.root),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "detectors": [{"name": d.name, "weight": d.weight, "options": d.opt} for d in scanner.detectors],
        "problems": scanner.problems,
        "notes": scanner.notes,
        "summary": {
            "score": total.score,
            "files": total.files,
            "scored_files": total.scored_files,
            "loc": total.loc,
            "flagged_files": total.flagged_files,
            "flagged_loc_share": total.flagged_share,
        },
        "files": files,
    }


def write(scanner: Scanner, skip: Callable[[str], bool], target: Path | None = None) -> Path:
    if target is None:
        target = STATE_DIR / "reports" / f"{scanner.root.name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    target = target.expanduser().resolve()
    if is_inside(target, scanner.root):
        raise ValueError(f"отчёт нельзя писать внутрь проверяемого репозитория: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build(scanner, skip), ensure_ascii=False, indent=1))
    return target
