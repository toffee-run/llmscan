"""Command line entry points: `llmscan` (scan a repository) and `llmscan-eval` (measure detectors)."""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import report
from .detectors import build_detectors, load_detectors
from .engine import Scanner
from .scoring import FLAG_THRESHOLD
from .storage import Exclusions, FindingCache


def _add_detector_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--detectors", help="список через запятую (по умолчанию все, кроме экспериментальных)")
    parser.add_argument("--set", dest="settings", action="append", default=[], metavar="DETECTOR.OPTION=VALUE",
                        help="настройка детектора, например lm.device=cpu или lm.weight=0.5")
    parser.add_argument("--no-cache", action="store_true", help="не читать и не писать кэш результатов")


def _detectors(args: argparse.Namespace):
    names = [n.strip() for n in args.detectors.split(",")] if args.detectors else None
    return build_detectors(names, args.settings)


def _list_detectors(console: Console) -> None:
    table = Table(box=None, padding=(0, 2))
    for column in ("имя", "вес", "по умолчанию", "настройки", "описание"):
        table.add_column(column)
    for name, cls in load_detectors().items():
        options = ", ".join(f"{k}={v}" for k, v in cls.options.items())
        table.add_row(name, f"{cls.weight:g}", "да" if cls.enabled_by_default else "нет", options, cls.description)
    console.print(table)


def _print_summary(console: Console, scanner: Scanner, skip, top: int) -> None:
    total = scanner.summary(skip)
    score = "нет данных" if total.score is None else f"{total.score:.0%}"
    console.print(f"[bold]{scanner.root}[/bold]")
    console.print(
        f"оценка LLM: [bold]{score}[/bold] · файлов {total.files} · строк {total.loc} · "
        f"≥{FLAG_THRESHOLD:.0%}: {total.flagged_files} файл(ов), {total.flagged_share:.0%} строк"
    )
    for note in scanner.notes:
        console.print(f"• {note}")
    for name, problem in scanner.problems.items():
        console.print(f"[yellow]! {name}: {problem}[/yellow]")
    table = Table(box=None, padding=(0, 2))
    table.add_column("итог", justify="right")
    table.add_column("увер.", justify="right")
    for detector in scanner.detectors:
        table.add_column(detector.name, justify="right")
    table.add_column("строк", justify="right")
    table.add_column("файл")
    ranked = sorted((rel for rel in scanner.reports if not skip(rel)), key=lambda rel: -scanner.rank(rel))
    for rel in ranked[:top]:
        entry = scanner.reports[rel]
        cells = []
        for detector in scanner.detectors:
            finding = entry.findings.get(detector.name)
            cells.append("—" if finding is None or finding.score is None else f"{finding.score:.0%}")
        combined = scanner.score(rel)
        certainty = "—" if combined is None else f"{scanner.confidence(rel):.0%}"
        table.add_row("—" if combined is None else f"{combined:.0%}", certainty, *cells, str(entry.loc), rel)
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llmscan", description="Оценивает, какая часть репозитория написана LLM. Репозиторий только читается.")
    parser.add_argument("path", nargs="?", type=Path, help="путь к репозиторию")
    _add_detector_arguments(parser)
    parser.add_argument("--exclude", action="append", default=[], metavar="GLOB", help="исключить файлы по маске (можно несколько раз)")
    parser.add_argument("--no-gitignore", action="store_true", help="не учитывать .gitignore")
    parser.add_argument("--include-configs", action="store_true", help="не пропускать типовые конфиги (*.config.*, .eslintrc.js, env.d.ts…)")
    parser.add_argument("--no-tui", action="store_true", help="напечатать итог и выйти")
    parser.add_argument("--json", type=Path, metavar="FILE", help="сохранить полный отчёт (не внутри репозитория)")
    parser.add_argument("--top", type=int, default=25, help="сколько файлов показать в режиме --no-tui")
    parser.add_argument("--list-detectors", action="store_true")
    args = parser.parse_args(argv)
    console = Console()

    if args.list_detectors:
        _list_detectors(console)
        return 0
    if args.path is None:
        parser.error("нужен путь к репозиторию")
    root = args.path.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"не каталог: {root}")
    try:
        detectors = _detectors(args)
    except ValueError as exc:
        parser.error(str(exc))

    scanner = Scanner(root, detectors, None if args.no_cache else FindingCache(), use_gitignore=not args.no_gitignore,
                      skip_configs=not args.include_configs)
    scanner.discover()
    if not scanner.files:
        console.print("В каталоге нет файлов с исходным кодом.")
        return 1
    headless = args.no_tui or args.json is not None
    # Exclusions made in the TUI are remembered per repository; headless runs start clean.
    exclusions = Exclusions(root, persist=not headless)
    for rel in scanner.files:
        if any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(rel.rsplit("/", 1)[-1], g) for g in args.exclude):
            exclusions.paths.add(rel)
    skip = exclusions.__contains__

    try:
        if headless:
            def progress(detector, done, total, rel):
                if sys.stderr.isatty():
                    print(f"\r\033[K{detector.name}: {done}/{total}", end="", file=sys.stderr, flush=True)

            scanner.run(skip, progress)
            if sys.stderr.isatty():
                print("\r\033[K", end="", file=sys.stderr)
            _print_summary(console, scanner, skip, args.top)
            if args.json:
                try:
                    console.print(f"отчёт: {report.write(scanner, skip, args.json)}")
                except ValueError as exc:
                    console.print(f"[red]{exc}[/red]")
                    return 2
        else:
            from .tui import ScanApp

            ScanApp(scanner, exclusions).run()
    finally:
        scanner.close()
    return 0


def _auc(negatives: list[float], positives: list[float]) -> float:
    """Probability that a random LLM file scores above a random human file."""
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def eval_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llmscan-eval",
        description="Проверяет детекторы на ваших данных: каталоги с заведомо человеческим и заведомо LLM-кодом.",
    )
    parser.add_argument("--human", action="append", required=True, type=Path, help="каталог с человеческим кодом")
    parser.add_argument("--llm", action="append", required=True, type=Path, help="каталог с кодом от LLM")
    _add_detector_arguments(parser)
    args = parser.parse_args(argv)
    console = Console()
    try:
        detectors = [d for d in _detectors(args) if d.cacheable]
    except ValueError as exc:
        parser.error(str(exc))
    cache = None if args.no_cache else FindingCache()

    scores: dict[str, dict[str, list[float]]] = {"human": {}, "llm": {}}
    counts = {}
    for label, roots in (("human", args.human), ("llm", args.llm)):
        counts[label] = 0
        for root in roots:
            scanner = Scanner(root.expanduser(), detectors, cache)
            scanner.discover()
            scanner.run()
            counts[label] += len(scanner.files)
            for rel, entry in scanner.reports.items():
                for name, finding in entry.findings.items():
                    if finding.score is not None:
                        scores[label].setdefault(name, []).append(finding.score)
                if (combined := scanner.score(rel)) is not None:
                    scores[label].setdefault("ИТОГ", []).append(combined)
    for detector in detectors:
        detector.close()

    console.print(f"человеческих файлов: {counts['human']}, LLM-файлов: {counts['llm']}")
    table = Table(box=None, padding=(0, 2))
    for column in ("детектор", "AUC", "средн. человек", "средн. LLM", f"ложных ≥{FLAG_THRESHOLD:.0%}", f"найдено ≥{FLAG_THRESHOLD:.0%}"):
        table.add_column(column, justify="right")
    for name in [*(d.name for d in detectors), "ИТОГ"]:
        human, llm = scores["human"].get(name, []), scores["llm"].get(name, [])
        if not human or not llm:
            table.add_row(name, "—", "—", "—", "—", "—")
            continue
        table.add_row(
            name, f"{_auc(human, llm):.2f}", f"{sum(human) / len(human):.2f}", f"{sum(llm) / len(llm):.2f}",
            f"{sum(s >= FLAG_THRESHOLD for s in human) / len(human):.0%}", f"{sum(s >= FLAG_THRESHOLD for s in llm) / len(llm):.0%}",
        )
    console.print(table)
    console.print("AUC 0.5 — детектор не лучше монетки, 1.0 — идеально разделяет.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
