"""The scan loop: run every detector over every file, cheapest detector first."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .detectors import Detector
from .models import FileReport, SourceFile
from .repo import load_repo
from .scoring import Aggregate, aggregate, combine, confidence, shrunk
from .storage import FindingCache

# (detector, files done, files total, path just finished or None)
Progress = Callable[[Detector, int, int, str | None], None]


class Scanner:
    def __init__(self, root: Path, detectors: list[Detector], cache: FindingCache | None = None,
                 use_gitignore: bool = True, skip_configs: bool = True) -> None:
        self.root = root.resolve()
        self.detectors = detectors
        self.cache = cache
        self.use_gitignore = use_gitignore
        self.skip_configs = skip_configs
        self.files: dict[str, SourceFile] = {}
        self.reports: dict[str, FileReport] = {}
        self.problems: dict[str, str] = {}
        self.notes: list[str] = []
        self._done: set[tuple[str, str]] = set()

    @property
    def weights(self) -> dict[str, float]:
        return {d.name: d.weight for d in self.detectors}

    def discover(self) -> None:
        for file in load_repo(self.root, self.use_gitignore, self.skip_configs):
            self.files[file.rel] = file
            self.reports[file.rel] = FileReport(rel=file.rel, language=file.language, loc=file.loc)

    def score(self, rel: str) -> float | None:
        return combine(self.reports[rel].findings, self.weights)

    def confidence(self, rel: str) -> float:
        return confidence(self.reports[rel].findings, self.weights)

    def rank(self, rel: str) -> float:
        """Sort key for "most suspicious first": unsure scores sink towards the middle."""
        return shrunk(self.score(rel), self.confidence(rel))

    def summary(self, skip: Callable[[str], bool] = lambda rel: False, prefix: str = "") -> Aggregate:
        chosen = (r for rel, r in self.reports.items() if rel.startswith(prefix) and not skip(rel))
        return aggregate(chosen, self.weights)

    def run(
        self,
        skip: Callable[[str], bool] = lambda rel: False,
        on_progress: Progress | None = None,
        should_stop: Callable[[], bool] = lambda: False,
    ) -> None:
        """Fill in missing findings. Safe to call again after files are re-included."""
        notes: list[str] = []
        for detector in self.detectors:
            if should_stop():
                return
            if reason := detector.unavailable_reason():
                self.problems[detector.name] = reason
                continue
            key = detector.cache_key()
            pending = [f for rel, f in self.files.items() if (detector.name, rel) not in self._done and not skip(rel)]
            todo: list[SourceFile] = []
            for file in pending:
                cached = self.cache.get(key, file.sha) if self.cache and detector.cacheable else None
                if self.cache and detector.cacheable and not FindingCache.is_missing(cached):
                    self._record(detector, file, cached)
                else:
                    todo.append(file)
            total = len(pending)
            done = total - len(todo)
            if on_progress:
                on_progress(detector, done, total, None)
            # Models are loaded only when there is real work left after the cache.
            if todo or not detector.cacheable:
                try:
                    detector.prepare(self.root)
                except Exception as exc:  # noqa: BLE001 - a broken detector must not kill the scan
                    self.problems[detector.name] = f"{type(exc).__name__}: {exc}"
                    continue
            failures = 0
            for file in todo:
                if should_stop():
                    return
                if skip(file.rel):
                    continue
                try:
                    finding = detector.analyze(file)
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    self.problems[detector.name] = f"{failures} файл(ов) с ошибкой, последняя: {type(exc).__name__}: {exc}"
                    finding = None
                else:
                    if self.cache and detector.cacheable:
                        self.cache.put(key, file.sha, finding)
                self._record(detector, file, finding)
                done += 1
                if on_progress:
                    on_progress(detector, done, total, file.rel)
            notes.extend(detector.repo_notes())
        self.notes = notes

    def _record(self, detector: Detector, file: SourceFile, finding) -> None:
        self._done.add((detector.name, file.rel))
        if finding is not None:
            self.reports[file.rel].findings[detector.name] = finding

    def close(self) -> None:
        for detector in self.detectors:
            detector.close()
