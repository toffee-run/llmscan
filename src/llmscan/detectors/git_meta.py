"""Authorship trails left in git history and in agent config files.

History is read with `git --no-optional-locks log`, which never takes a lock or
refreshes the index, so the repository stays byte-for-byte untouched.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from ..models import Evidence, Finding, SourceFile
from .base import Detector, register

AI_MARKERS = re.compile(
    r"co-authored-by:.*\b(claude|copilot|cursor|codex|chatgpt|gemini|devin|aider|openhands|jules|windsurf|amp|cline)\b"
    r"|generated with \[?claude code|noreply@anthropic\.com|cursoragent@cursor\.com|aider \(https://aider\.chat\)"
    r"|\bcopilot-swe-agent\b|devin-ai-integration|\bgoogle-labs-jules\b|chatgpt-codex-connector|🤖 generated with",
    re.IGNORECASE,
)

AGENT_FILES = (
    "CLAUDE.md", "AGENTS.md", "GEMINI.md", ".cursorrules", ".cursor", ".windsurfrules", ".clinerules",
    ".aider.conf.yml", ".aider.chat.history.md", ".github/copilot-instructions.md", ".claude", ".codex",
)

_SEP = "\x1e"


@register
class GitMeta(Detector):
    name = "git"
    title = "История git"
    description = "Коммиты с подписью ИИ-ассистента (Co-Authored-By: Claude, Copilot, Cursor…) и конфиги агентов."
    weight = 3.0
    cost = 1
    cacheable = False
    options = {"max_commits": 20000}

    def __init__(self, **overrides) -> None:
        super().__init__(**overrides)
        self._touched: dict[str, int] = defaultdict(int)
        self._marked: dict[str, int] = defaultdict(int)
        self._examples: dict[str, str] = {}
        self._notes: list[str] = []
        self._has_history = False

    def prepare(self, root: Path) -> None:
        self._touched.clear()
        self._marked.clear()
        self._examples.clear()
        present = [name for name in AGENT_FILES if (root / name).exists()]
        self._notes = [f"в репозитории есть файлы ИИ-агентов: {', '.join(present)}"] if present else []
        try:
            out = subprocess.run(
                ["git", "--no-optional-locks", "-C", str(root), "log", f"-n{self.opt['max_commits']}",
                 "--no-renames", "--name-only", f"--format={_SEP}%h%x00%an <%ae>%x00%B%x00"],
                capture_output=True, text=True, errors="replace", timeout=120,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            )
        except (OSError, subprocess.TimeoutExpired):
            return
        if out.returncode != 0:
            return
        # `git log` prints paths relative to the repository top level, which may be above `root`.
        top = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(root), "rev-parse", "--show-prefix"],
            capture_output=True, text=True, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        ).stdout.strip()
        commits = marked_commits = 0
        for chunk in out.stdout.split(_SEP)[1:]:
            parts = chunk.split("\0")
            if len(parts) < 4:
                continue
            sha, author, body, names = parts[0], parts[1], parts[2], parts[3]
            match = AI_MARKERS.search(author) or AI_MARKERS.search(body)
            commits += 1
            marked_commits += bool(match)
            for name in names.splitlines():
                if not name or not name.startswith(top):
                    continue
                rel = name[len(top):]
                self._touched[rel] += 1
                if match:
                    self._marked[rel] += 1
                    self._examples.setdefault(rel, f"{sha}: {match.group(0).strip()[:70]}")
        self._has_history = commits > 0
        if commits:
            self._notes.append(f"коммитов с подписью ИИ: {marked_commits} из {commits} ({marked_commits / commits:.0%})")

    def analyze(self, file: SourceFile) -> Finding | None:
        touched = self._touched.get(file.rel, 0)
        marked = self._marked.get(file.rel, 0)
        if not self._has_history or not marked:
            # No trailer proves nothing: most people who use an assistant do not sign commits.
            return None
        share = marked / touched
        return Finding(
            score=0.7 + 0.3 * share,
            confidence=1.0,
            summary=f"{marked} из {touched} коммитов файла подписаны ИИ",
            evidence=[Evidence(f"например {self._examples[file.rel]}")],
            metrics={"commits": touched, "ai_commits": marked},
        )

    def repo_notes(self) -> list[str]:
        return list(self._notes)
