import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from llmscan import report
from llmscan.detectors import Detector, build_detectors, register
from llmscan.engine import Scanner
from llmscan.models import FileReport, Finding
from llmscan.repo import load_repo
from llmscan.scoring import aggregate, combine, shrunk
from llmscan.storage import Exclusions

LLM_STYLE = '''# Step 1: Initialize the configuration
config = {}
# Create the client
client = make()  # ✅ done
# ... rest of the code remains unchanged
print("🚀 Starting")
'''
HUMAN_STYLE = '''import os
# TODO: fix this hack
# x = compute(1)
def f(a):
    return os.path.join(a, "b")
'''


@register
class Marker(Detector):
    """Deterministic stand-in for a real detector."""

    name = "marker"

    def analyze(self, file):
        return Finding(score=0.9 if "🚀" in file.text else 0.1)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "node_modules" / "dep").mkdir(parents=True)
    (root / "pkg" / "llm.py").write_text(LLM_STYLE)
    (root / "pkg" / "human.py").write_text(HUMAN_STYLE)
    (root / "pkg" / "ignored.py").write_text("x = 1\ny = 2\nz = 3\n")
    (root / "node_modules" / "dep" / "index.js").write_text("var a = 1;\n")
    (root / "image.png").write_bytes(b"\x89PNG\0\0")
    (root / ".gitignore").write_text("ignored.py\n")
    (root / "eslint.config.mjs").write_text("export default [];\n")
    return root


def snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(root.rglob("*")) if p.is_file()}


def test_discovery_respects_gitignore_and_skips_vendored(repo: Path) -> None:
    assert [f.rel for f in load_repo(repo)] == ["pkg/human.py", "pkg/llm.py"]
    assert "pkg/ignored.py" in [f.rel for f in load_repo(repo, use_gitignore=False)]
    assert "eslint.config.mjs" in [f.rel for f in load_repo(repo, skip_configs=False)]


def test_summary_is_line_weighted_and_honours_exclusions(repo: Path) -> None:
    scanner = Scanner(repo, build_detectors(["marker"]))
    scanner.discover()
    scanner.run()
    total = scanner.summary()
    assert (total.files, total.flagged_files) == (2, 1)
    assert total.score == pytest.approx((0.9 * 6 + 0.1 * 5) / 11)
    assert scanner.summary(lambda rel: rel == "pkg/llm.py").score == pytest.approx(0.1)


def test_git_detector_flags_signed_commits_and_leaves_repo_untouched(repo: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True, capture_output=True)

    git("init", "-q")
    git("add", "pkg/human.py")
    git("commit", "-qm", "by hand")
    git("add", "pkg/llm.py")
    git("commit", "-qm", "feature\n\nCo-Authored-By: Claude <noreply@anthropic.com>")
    before = snapshot(repo)
    scanner = Scanner(repo, build_detectors(["marker", "git"]))
    scanner.discover()
    scanner.run()
    assert snapshot(repo) == before
    assert scanner.reports["pkg/llm.py"].findings["git"].score == 1.0
    assert "git" not in scanner.reports["pkg/human.py"].findings
    assert any("1 из 2" in note for note in scanner.notes)


def test_combine_ignores_abstentions() -> None:
    findings = {"a": Finding(score=0.8), "b": Finding(score=None), "c": Finding(score=0.2, confidence=0.0)}
    assert combine(findings, {"a": 1, "b": 5, "c": 1}) == pytest.approx(0.8)
    assert combine({"b": Finding(score=None)}, {}) is None


def test_unsure_scores_weigh_less_and_are_not_flagged() -> None:
    weights = {"lm": 1.0}
    reports = [
        FileReport("noisy.ts", "typescript", 100, {"lm": Finding(score=0.95, confidence=0.1)}),
        FileReport("clear.ts", "typescript", 100, {"lm": Finding(score=0.05, confidence=0.9)}),
    ]
    total = aggregate(reports, weights)
    assert total.score == pytest.approx((0.95 * 0.1 + 0.05 * 0.9) / 1.0)
    assert total.flagged_files == 0
    assert shrunk(0.95, 0.1) < shrunk(0.8, 0.9)


def test_report_refuses_to_write_inside_repo(repo: Path, tmp_path: Path) -> None:
    scanner = Scanner(repo, build_detectors(["marker"]))
    scanner.discover()
    scanner.run()
    with pytest.raises(ValueError):
        report.write(scanner, lambda rel: False, repo / "out.json")
    data = json.loads(report.write(scanner, lambda rel: False, tmp_path / "out.json").read_text())
    assert data["files"][0]["path"] == "pkg/llm.py"


def test_new_detector_only_needs_register(repo: Path) -> None:
    @register
    class AlwaysHalf(Detector):
        name = "half"
        options = {"value": 0.5}

        def analyze(self, file):
            return Finding(score=self.opt["value"])

    scanner = Scanner(repo, build_detectors(["half"], ["half.value=0.9", "half.weight=2"]))
    scanner.discover()
    scanner.run()
    assert scanner.score("pkg/human.py") == pytest.approx(0.9)
    assert scanner.weights == {"half": 2.0}


def test_tui_excludes_file_directory_and_pattern(repo: Path, tmp_path: Path, monkeypatch) -> None:
    from llmscan import storage
    from llmscan.tui import ScanApp

    monkeypatch.setattr(storage, "STATE_DIR", tmp_path / "state")
    before = snapshot(repo)
    scanner = Scanner(repo, build_detectors(["marker"]))
    scanner.discover()
    exclusions = Exclusions(repo)

    async def drive() -> None:
        app = ScanApp(scanner, exclusions)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause(0.8)
            assert scanner.score("pkg/llm.py") is not None
            await pilot.press("down", "space")  # the pkg/ directory
            assert exclusions.paths == {"pkg/human.py", "pkg/llm.py"}
            await pilot.press("space")
            assert exclusions.paths == set()
            await pilot.press("down", "space")  # first file
            assert exclusions.paths == {"pkg/human.py"}
            await pilot.press("a", "x")
            await pilot.pause()
            await pilot.press("l", "l", "m", "asterisk", "enter")
            await pilot.pause()
            assert exclusions.paths == {"pkg/llm.py"}
            await pilot.press("s", "e")
            await pilot.pause(0.5)

    asyncio.run(drive())
    assert Exclusions(repo).paths == {"pkg/llm.py"}
    assert snapshot(repo) == before
