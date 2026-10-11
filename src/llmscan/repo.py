"""Read-only discovery of the files worth analysing.

Nothing here (or anywhere else in llmscan) opens a repository file for writing,
and git is never invoked from this module: .gitignore files are parsed by hand
so that walking the tree cannot touch the index.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
from pathlib import Path
from typing import Iterator

from pathspec import GitIgnoreSpec

from .models import SourceFile

MAX_FILE_BYTES = 400_000

LANGUAGES = {
    ".py": "python", ".pyi": "python", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript", ".go": "go", ".rs": "rust",
    ".java": "java", ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala", ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".cs": "csharp",
    ".rb": "ruby", ".php": "php", ".swift": "swift", ".m": "objc", ".mm": "objc", ".lua": "lua",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".fish": "shell", ".ps1": "powershell",
    ".sql": "sql", ".r": "r", ".jl": "julia", ".dart": "dart", ".ex": "elixir", ".exs": "elixir",
    ".erl": "erlang", ".hs": "haskell", ".ml": "ocaml", ".clj": "clojure", ".zig": "zig", ".nix": "nix",
    ".vue": "vue", ".svelte": "svelte", ".html": "html", ".css": "css", ".scss": "css",
    ".tf": "terraform", ".proto": "protobuf", ".prisma": "prisma", ".graphql": "graphql", ".gql": "graphql", ".md": "markdown", ".mdx": "markdown", ".rst": "markdown",
}
FILENAMES = {"Dockerfile": "docker", "Makefile": "make", "Justfile": "make"}

# Never interesting, even when a repository forgets to ignore them.
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "vendor", "third_party", ".venv", "venv", "env",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "dist", "build", "target",
    ".next", ".nuxt", ".idea", ".vscode", "site-packages", ".gradle", ".terraform", "coverage",
}
SKIP_GLOBS = ("*.min.js", "*.min.css", "*.lock", "*-lock.json", "*.pb.go", "*_pb2.py", "*.generated.*", "*.snap")

# Tool configuration: copied from docs and templates by people and models alike, so it says nothing about authorship.
CONFIG_GLOBS = (
    "*.config.*", "*.conf.js", "*.conf.ts", ".*rc.js", ".*rc.cjs", ".*rc.mjs", ".*rc.ts", "*env.d.ts",
    "gulpfile.*", "Gruntfile.*", "webpack.*.js", "*.gradle.kts", "jest.setup.*", "vitest.setup.*", "vitest.workspace.*",
)


def language_of(path: Path) -> str | None:
    return FILENAMES.get(path.name) or LANGUAGES.get(path.suffix.lower())


def _load_ignore(directory: Path) -> GitIgnoreSpec | None:
    ignore = directory / ".gitignore"
    try:
        lines = ignore.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    return GitIgnoreSpec.from_lines(lines)


def _ignored(specs: list[tuple[Path, GitIgnoreSpec]], path: Path, is_dir: bool) -> bool:
    """Apply nested .gitignore files outermost-first; the last match wins, as in git."""
    verdict = False
    for base, spec in specs:
        rel = path.relative_to(base).as_posix() + ("/" if is_dir else "")
        result = spec.check_file(rel)
        if result.include is not None:
            verdict = result.include
    return verdict


def iter_paths(root: Path, use_gitignore: bool = True, skip_configs: bool = True) -> Iterator[Path]:
    """Yield candidate source files under root, sorted, without following symlinks."""

    skipped = SKIP_GLOBS + CONFIG_GLOBS if skip_configs else SKIP_GLOBS

    def walk(directory: Path, specs: list[tuple[Path, GitIgnoreSpec]]) -> Iterator[Path]:
        if use_gitignore and (spec := _load_ignore(directory)) is not None:
            specs = [*specs, (directory, spec)]
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError:
            return
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False):
                if entry.name in SKIP_DIRS or _ignored(specs, path, True):
                    continue
                yield from walk(path, specs)
            elif entry.is_file(follow_symlinks=False):
                if language_of(path) is None or any(fnmatch.fnmatch(entry.name, g) for g in skipped):
                    continue
                if _ignored(specs, path, False):
                    continue
                yield path

    yield from walk(root, [])


def read_source(root: Path, path: Path) -> SourceFile | None:
    """Read one file; returns None for binary, oversized, empty or unreadable files."""
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        raw = path.read_bytes()
    except OSError:
        return None
    if not raw.strip() or b"\0" in raw[:8192]:
        return None
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    # Minified or machine-generated blobs say nothing about authorship.
    if lines and max(len(line) for line in lines) > 2000:
        return None
    return SourceFile(
        path=path,
        rel=path.relative_to(root).as_posix(),
        text=text,
        language=language_of(path) or "text",
        sha=hashlib.sha1(raw).hexdigest(),
    )


def load_repo(root: Path, use_gitignore: bool = True, skip_configs: bool = True) -> list[SourceFile]:
    files = (read_source(root, p) for p in iter_paths(root, use_gitignore, skip_configs))
    return [f for f in files if f is not None]


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
