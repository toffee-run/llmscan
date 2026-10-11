"""The detector contract and the registry.

Adding a feature means adding a detector: drop a module next to this one (or in
~/.config/llmscan/detectors/, or ship it as a `llmscan.detectors` entry point),
subclass Detector, decorate it with @register. Nothing else needs editing: the
CLI, the cache, the scoring and the TUI pick it up from the registry.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, ClassVar

from ..models import Finding, SourceFile

USER_PLUGIN_DIR = Path.home() / ".config" / "llmscan" / "detectors"

_REGISTRY: dict[str, type["Detector"]] = {}


class Detector:
    """Base class for every signal llmscan can compute about a file."""

    name: ClassVar[str]
    title: ClassVar[str] = ""
    description: ClassVar[str] = ""
    # Relative trust in this detector when scores are combined.
    weight: ClassVar[float] = 1.0
    # Bump to invalidate cached findings after changing the logic.
    version: ClassVar[int] = 1
    # Cheap detectors run first so the UI fills in quickly.
    cost: ClassVar[int] = 0
    # False when the finding depends on more than the file's bytes (e.g. git history).
    cacheable: ClassVar[bool] = True
    # Experimental detectors set this to False and run only when named in --detectors.
    enabled_by_default: ClassVar[bool] = True
    # Defaults for user-tunable options, overridable with `--set name.key=value`.
    options: ClassVar[dict[str, Any]] = {}

    def __init__(self, **overrides: Any) -> None:
        unknown = set(overrides) - set(self.options)
        if unknown:
            raise ValueError(f"{self.name}: unknown option(s) {', '.join(sorted(unknown))}")
        self.opt = {**self.options, **overrides}

    def unavailable_reason(self) -> str | None:
        """Return why this detector cannot run here, or None when it can."""
        return None

    def prepare(self, root: Path) -> None:
        """Called once per scan before analyze(); load models, read history, etc."""

    def analyze(self, file: SourceFile) -> Finding | None:
        """Score one file. Return None (or score=None) to abstain."""
        raise NotImplementedError

    def repo_notes(self) -> list[str]:
        """Repository-level observations shown in the summary."""
        return []

    def close(self) -> None:
        """Release resources taken in prepare()."""

    def cache_key(self) -> str:
        options = ",".join(f"{k}={self.opt[k]}" for k in sorted(self.opt))
        return f"{self.name}:{self.version}:{options}"


def register(cls: type[Detector]) -> type[Detector]:
    _REGISTRY[cls.name] = cls
    return cls


def _coerce(default: Any, raw: str) -> Any:
    if isinstance(default, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(default, (int, float)):
        return type(default)(raw)
    return raw


def load_detectors() -> dict[str, type[Detector]]:
    """Import every detector module so that @register runs, then return the registry."""
    package = importlib.import_module(__package__)
    for module in pkgutil.iter_modules(package.__path__):
        if module.name != "base":
            importlib.import_module(f"{__package__}.{module.name}")
    for entry in entry_points(group="llmscan.detectors"):
        entry.load()
    if USER_PLUGIN_DIR.is_dir():
        for path in sorted(USER_PLUGIN_DIR.glob("*.py")):
            module_name = f"llmscan_user_{path.stem}"
            if module_name in sys.modules:
                continue
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
    return dict(_REGISTRY)


def build_detectors(names: list[str] | None = None, settings: list[str] | None = None) -> list[Detector]:
    """Instantiate detectors, applying `detector.key=value` overrides."""
    registry = load_detectors()
    chosen = names or [name for name, cls in registry.items() if cls.enabled_by_default]
    missing = [n for n in chosen if n not in registry]
    if missing:
        raise ValueError(f"unknown detector(s): {', '.join(missing)}; available: {', '.join(registry)}")
    overrides: dict[str, dict[str, Any]] = {}
    for item in settings or []:
        target, _, raw = item.partition("=")
        detector, _, key = target.partition(".")
        if detector not in registry or (key != "weight" and key not in registry[detector].options):
            raise ValueError(f"bad --set '{item}': expected detector.option=value")
        default = registry[detector].weight if key == "weight" else registry[detector].options[key]
        overrides.setdefault(detector, {})[key] = _coerce(default, raw)
    detectors = []
    for name in chosen:
        options = dict(overrides.get(name, {}))
        weight = options.pop("weight", None)
        detector = registry[name](**options)
        if weight is not None:
            # Shadows the class attribute for this run only.
            detector.weight = weight
        detectors.append(detector)
    return sorted(detectors, key=lambda d: d.cost)
