"""The terminal UI: a file tree with scores on the left, the reasons on the right."""

from __future__ import annotations

import fnmatch
import threading
from collections import defaultdict

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Footer, Input, Label, Static, Tree
from textual.widgets.tree import TreeNode

from . import report
from .detectors import Detector
from .engine import Scanner
from .scoring import FLAG_THRESHOLD, MIN_CONFIDENCE
from .storage import Exclusions

SORTS = {"name": "по имени", "score": "по оценке", "loc": "по размеру"}
BAR_WIDTH = 8


def score_style(score: float | None) -> str:
    if score is None:
        return "dim"
    if score >= FLAG_THRESHOLD:
        return "bold red"
    if score >= 0.4:
        return "yellow"
    return "green"


def score_text(score: float | None, unsure: bool = False) -> Text:
    """A bar and a percentage; scores the detectors are unsure about are dimmed and marked with `?`."""
    if score is None:
        return Text(" " * BAR_WIDTH + "   —  ", style="dim")
    filled = round(score * BAR_WIDTH)
    text = Text("█" * filled + "░" * (BAR_WIDTH - filled), style="dim" if unsure else score_style(score))
    text.append(f" {score:4.0%}" + ("?" if unsure else " "), style="dim" if unsure else "")
    return text


def human_count(value: int) -> str:
    return f"{value / 1000:.1f}k" if value >= 1000 else str(value)


class FileTree(Tree[str]):
    # Tree binds space to expand/collapse; here it is the exclusion toggle, enter still expands.
    BINDINGS = [Binding("space", "toggle_exclude", "Искл./вкл.")]

    def action_toggle_exclude(self) -> None:
        if self.cursor_node is not None:
            self.app.toggle_exclude(self.cursor_node)  # type: ignore[attr-defined]


class PatternScreen(ModalScreen[str | None]):
    """Ask for a glob such as `tests/*` or `*.md`."""

    BINDINGS = [Binding("escape", "dismiss(None)", "Отмена")]
    DEFAULT_CSS = """
    PatternScreen { align: center middle; }
    PatternScreen > Vertical { width: 70; height: auto; border: round $accent; padding: 1 2; background: $surface; }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Исключить файлы по маске (например tests/*, *.md, docs/**). Enter — применить, Esc — отмена.")
            yield Input(placeholder="маска")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)


class ScanApp(App[None]):
    TITLE = "llmscan"
    CSS = """
    #summary { height: auto; padding: 0 1; background: $boost; }
    #status { height: 1; padding: 0 1; color: $text-muted; }
    FileTree { width: 55%; border-right: solid $panel; }
    #detail-pane { width: 45%; padding: 0 1; }
    """
    BINDINGS = [
        Binding("x", "pattern", "По маске"),
        Binding("a", "include_all", "Вернуть все"),
        Binding("s", "sort", "Сортировка"),
        Binding("e", "export", "Экспорт JSON"),
        Binding("q", "quit", "Выход"),
    ]

    def __init__(self, scanner: Scanner, exclusions: Exclusions) -> None:
        super().__init__()
        self.scanner = scanner
        self.exclusions = exclusions
        self.sort_mode = "name"
        self._tree_nodes: dict[str, TreeNode[str]] = {}
        self._members: dict[str, list[str]] = defaultdict(list)
        # Direct children of each directory; a dict keeps insertion order and drops duplicates.
        self._kids: dict[str, dict[str, None]] = defaultdict(dict)
        self._resort_wanted = False
        self._dirty_keys: set[str] = set()
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._scan_wanted = True
        self._scanning = False
        self._status_text = "поиск файлов…"

    # ---- layout

    def compose(self) -> ComposeResult:
        yield Static(id="summary")
        with Horizontal():
            yield FileTree(self.scanner.root.name, data="", id="tree")
            with VerticalScroll(id="detail-pane"):
                yield Static(id="detail")
        yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        for rel in self.scanner.files:
            parts = rel.split("/")
            for depth in range(len(parts)):
                parent = "/".join(parts[:depth])
                self._members[parent].append(rel)
                self._kids[parent]["/".join(parts[: depth + 1])] = None
        self._build_tree()
        self.set_interval(0.3, self._refresh_view)
        self.query_one(FileTree).focus()

    def on_unmount(self) -> None:
        self._stop_event.set()

    # ---- tree

    def _is_excluded(self, rel: str) -> bool:
        return rel in self.exclusions

    def _unsure(self, rel: str) -> bool:
        return self.scanner.confidence(rel) < MIN_CONFIDENCE

    def _sort_key(self, key: str):
        """Order of siblings: directories are ranked by their aggregate, like files."""
        is_file = key in self.scanner.files
        if self.sort_mode == "name":
            return (is_file, 0, key)
        total = None if is_file else self.scanner.summary(self._is_excluded, prefix=f"{key}/")
        if self.sort_mode == "score":
            value = self.scanner.rank(key) if is_file else (-1.0 if total.score is None else total.score)
        else:
            value = self.scanner.reports[key].loc if is_file else total.loc
        return (False, -value, key)

    def _build_tree(self) -> None:
        tree = self.query_one(FileTree)
        cursor = tree.cursor_node.data if tree.cursor_node is not None else None
        collapsed = {key for key, node in self._tree_nodes.items() if node.allow_expand and not node.is_expanded}
        tree.clear()
        self._tree_nodes = {"": tree.root}

        def fill(parent: TreeNode[str], parent_key: str) -> None:
            for key in sorted(self._kids[parent_key], key=self._sort_key):
                name = key.rsplit("/", 1)[-1]
                if key in self.scanner.files:
                    self._tree_nodes[key] = parent.add_leaf(name, data=key)
                else:
                    self._tree_nodes[key] = parent.add(name, data=key, expand=key not in collapsed)
                    fill(self._tree_nodes[key], key)

        fill(tree.root, "")
        tree.root.expand()
        for key in self._tree_nodes:
            self._relabel(key)
        if cursor and cursor in self._tree_nodes:
            # The rebuilt tree has new nodes; put the cursor back on the same path.
            self.call_after_refresh(tree.move_cursor, self._tree_nodes[cursor])

    def _relabel(self, key: str) -> None:
        node = self._tree_nodes.get(key)
        if node is None:
            return
        name = key.rsplit("/", 1)[-1] or self.scanner.root.name
        if key in self.scanner.files:
            excluded = self._is_excluded(key)
            score, loc, unsure = self.scanner.score(key), self.scanner.reports[key].loc, self._unsure(key)
        else:
            unsure = False
            members = self._members[key]
            excluded = bool(members) and all(self._is_excluded(m) for m in members)
            total = self.scanner.summary(self._is_excluded, prefix=f"{key}/" if key else "")
            score, loc, name = total.score, total.loc, f"{name}/"
        label = Text("✗ " if excluded else "✓ ", style="dim" if excluded else "green")
        if excluded:
            label.append(" " * (BAR_WIDTH + 6) + " " + name, style="dim strike")
        else:
            label.append_text(score_text(score, unsure))
            label.append(f" {name}")
            label.append(f"  {human_count(loc)}", style="dim")
        node.set_label(label)

    def _mark_dirty(self, rel: str) -> None:
        """Queue a file and its parent directories for a label refresh."""
        parts = rel.split("/")
        with self._state_lock:
            self._dirty_keys.update("/".join(parts[:depth]) for depth in range(len(parts) + 1))

    # ---- scanning

    def _refresh_view(self) -> None:
        if self._scan_wanted and not self._scanning:
            self._scan_wanted = False
            self._scanning = True
            self.run_worker(self._scan_worker, thread=True, group="scan")
        with self._state_lock:
            dirty, self._dirty_keys = self._dirty_keys, set()
        if self._resort_wanted:
            self._resort_wanted = False
            self._build_tree()
        else:
            for key in dirty:
                self._relabel(key)
        self._show_summary()
        self.query_one("#status", Static).update(f"{self._status_text}  · сортировка {SORTS[self.sort_mode]}")
        cursor = self.query_one(FileTree).cursor_node
        if cursor is not None and cursor.data in dirty:
            self._show_detail(cursor.data or "")

    def _scan_worker(self) -> None:
        def on_progress(detector: Detector, done: int, total: int, rel: str | None) -> None:
            device = getattr(detector, "device", None)
            suffix = f" [{device}]" if device and done else ""
            self._status_text = f"{detector.title or detector.name}{suffix}: {done}/{total}  {rel or ''}"
            if rel:
                self._mark_dirty(rel)

        try:
            self.scanner.run(self._is_excluded, on_progress, self._stop_event.is_set)
            problems = "; ".join(f"{name}: {text}" for name, text in self.scanner.problems.items())
            self._status_text = "готово" + (f" · проблемы — {problems}" if problems else "")
            # Scores arrived after the tree was ordered, so the order is stale.
            self._resort_wanted = self.sort_mode != "name"
        finally:
            with self._state_lock:
                self._dirty_keys.update(self._tree_nodes)
            self._scanning = False

    # ---- rendering

    def _show_summary(self) -> None:
        total = self.scanner.summary(self._is_excluded)
        line = Text(f"{self.scanner.root}  ", style="bold")
        line.append("оценка LLM ")
        line.append_text(score_text(total.score))
        line.append(
            f"   файлов {total.files} (исключено {len(self.exclusions.paths & self.scanner.files.keys())})"
            f" · строк {human_count(total.loc)}"
            f" · ≥{FLAG_THRESHOLD:.0%}: {total.flagged_files} файл(ов), {total.flagged_share:.0%} строк"
        )
        self.query_one("#summary", Static).update(line)

    def _show_detail(self, key: str) -> None:
        self.query_one("#detail", Static).update(self._file_detail(key) if key in self.scanner.files else self._dir_detail(key))

    def _file_detail(self, rel: str) -> RenderableType:
        entry = self.scanner.reports[rel]
        parts: list[RenderableType] = [Text(rel, style="bold")]
        head = Text(f"{entry.language} · {entry.loc} строк · итог ")
        head.append_text(score_text(self.scanner.score(rel), self._unsure(rel)))
        if entry.findings:
            head.append(f" уверенность {self.scanner.confidence(rel):.0%}", style="dim")
        if entry.findings and self._unsure(rel):
            head.append("\n? — оценка ненадёжна (мало кода или значение у порога), в «помечено» не входит", style="dim")
        if self._is_excluded(rel):
            head.append("  исключён из расчёта", style="dim")
        parts.append(head)
        for detector in self.scanner.detectors:
            finding = entry.findings.get(detector.name)
            title = Text(f"\n{detector.title or detector.name} ", style="bold underline")
            title.append(f"(вес {detector.weight:g})", style="dim")
            parts.append(title)
            if finding is None:
                parts.append(Text("  нет мнения / ещё не посчитан", style="dim"))
                continue
            line = Text("  ")
            line.append_text(score_text(finding.score))
            line.append(f"  уверенность {finding.confidence:.0%}", style="dim")
            parts.extend([line, Text(f"  {finding.summary}")])
            if finding.metrics:
                parts.append(Text("  " + "  ".join(f"{k}={v:g}" for k, v in finding.metrics.items()), style="dim"))
            for item in finding.evidence[:30]:
                where = "" if item.line is None else f"L{item.line}" + (f"–{item.end_line}" if item.end_line else "")
                row = Text(f"  {where:>10} ", style="cyan")
                if item.score is not None and 0 <= item.score <= 1 and item.end_line:
                    row.append(f"{item.score:4.0%} ", style=score_style(item.score))
                row.append(item.message)
                parts.append(row)
        return Group(*parts)

    def _dir_detail(self, key: str) -> RenderableType:
        prefix = f"{key}/" if key else ""
        total = self.scanner.summary(self._is_excluded, prefix=prefix)
        head = Text(f"{key or self.scanner.root.name}/\n", style="bold")
        head.append("итог ", style="not bold")
        head.append_text(score_text(total.score))
        head.append(
            f"\nфайлов {total.files}, с оценкой {total.scored_files}, строк {human_count(total.loc)}\n"
            f"помечено (≥{FLAG_THRESHOLD:.0%}): {total.flagged_files} файл(ов), {total.flagged_share:.0%} строк",
            style="not bold",
        )
        parts: list[RenderableType] = [head]
        if not key:
            notes = [*self.scanner.notes, *(f"{n}: {t}" for n, t in self.scanner.problems.items())]
            parts.extend(Text(f"• {note}", style="yellow") for note in notes)
            table = Table(title="Детекторы", title_justify="left", box=None, padding=(0, 1))
            for column in ("имя", "вес", "что делает"):
                table.add_column(column)
            for detector in self.scanner.detectors:
                table.add_row(detector.name, f"{detector.weight:g}", detector.description)
            parts.extend([Text(""), table])
        ranked = sorted(
            ((self.scanner.score(rel), rel) for rel in self._members[key] if not self._is_excluded(rel)),
            key=lambda pair: -self.scanner.rank(pair[1]),
        )
        top = Table(title="Самые подозрительные файлы", title_justify="left", box=None, padding=(0, 1))
        top.add_column("оценка")
        top.add_column("файл")
        for score, rel in ranked[:15]:
            top.add_row(score_text(score, self._unsure(rel)), rel[len(prefix):])
        parts.extend([Text(""), top])
        return Group(*parts)

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[str]) -> None:
        self._show_detail(event.node.data or "")

    # ---- actions

    def _set_excluded(self, rels: list[str], excluded: bool) -> None:
        if excluded:
            self.exclusions.paths.update(rels)
        else:
            self.exclusions.paths.difference_update(rels)
            # Files excluded before they were scanned still need their findings.
            self._scan_wanted = True
        self.exclusions.save()
        for rel in rels:
            self._mark_dirty(rel)

    def toggle_exclude(self, node: TreeNode[str]) -> None:
        key = node.data or ""
        rels = [key] if key in self.scanner.files else self._members[key]
        if rels:
            self._set_excluded(rels, not all(self._is_excluded(rel) for rel in rels))
            with self._state_lock:
                self._dirty_keys.add(key)

    def action_pattern(self) -> None:
        def apply(pattern: str | None) -> None:
            if not pattern:
                return
            matched = [
                rel for rel in self.scanner.files
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel.rsplit("/", 1)[-1], pattern)
            ]
            self._set_excluded(matched, True)
            self.notify(f"Исключено файлов: {len(matched)}")

        self.push_screen(PatternScreen(), apply)

    def action_include_all(self) -> None:
        self._set_excluded(list(self.exclusions.paths), False)

    def action_sort(self) -> None:
        modes = list(SORTS)
        self.sort_mode = modes[(modes.index(self.sort_mode) + 1) % len(modes)]
        self._build_tree()

    def action_export(self) -> None:
        path = report.write(self.scanner, self._is_excluded)
        self.notify(f"Отчёт: {path}", timeout=10)
