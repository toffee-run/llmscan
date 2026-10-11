"""Everything llmscan writes to disk. All of it lives under the user's home, never in the scanned repo."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path

from .models import Finding

CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "llmscan"
STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "llmscan"

_MISSING = object()


class FindingCache:
    """Findings keyed by detector configuration and file content hash."""

    def __init__(self, path: Path | None = None) -> None:
        path = path or CACHE_DIR / "findings.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS findings (key TEXT PRIMARY KEY, value TEXT)")

    def get(self, detector_key: str, sha: str) -> Finding | None | object:
        """Return the cached finding, None for a cached abstention, or MISSING."""
        with self._lock:
            row = self._db.execute("SELECT value FROM findings WHERE key = ?", (f"{detector_key}|{sha}",)).fetchone()
        if row is None:
            return _MISSING
        data = json.loads(row[0])
        return None if data is None else Finding.from_dict(data)

    def put(self, detector_key: str, sha: str, finding: Finding | None) -> None:
        value = json.dumps(None if finding is None else finding.to_dict())
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO findings VALUES (?, ?)", (f"{detector_key}|{sha}", value))
            self._db.commit()

    @staticmethod
    def is_missing(value: object) -> bool:
        return value is _MISSING


class Exclusions:
    """The set of files the user excluded for one repository, remembered between runs."""

    def __init__(self, root: Path, persist: bool = True) -> None:
        digest = hashlib.sha1(str(root.resolve()).encode()).hexdigest()[:16]
        self._path = STATE_DIR / "exclusions" / f"{digest}.json"
        self._root = root
        self._persist = persist
        self.paths: set[str] = set()
        if persist:
            try:
                self.paths = set(json.loads(self._path.read_text())["excluded"])
            except (OSError, ValueError, KeyError):
                pass

    def __contains__(self, rel: str) -> bool:
        return rel in self.paths

    def save(self) -> None:
        if not self._persist:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({"root": str(self._root.resolve()), "excluded": sorted(self.paths)}, indent=1))
