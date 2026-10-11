"""Binoculars-style detector running two small code LMs from Hugging Face.

Idea (Hans et al., 2024, "Spotting LLMs with Binoculars"): text written by an
LLM is unsurprising to another LLM *relative to* how surprising that LLM expects
it to be. The score is log-perplexity under the performer model divided by the
cross-perplexity between observer and performer. Lower means more machine-like.

The method was designed for prose. On the small code sample used while building
llmscan it separated human from LLM code barely better than chance (AUC ~0.54),
and the threshold is eyeballed, not fitted. Measure it with `llmscan-eval` and
tune `threshold`/`scale`/`weight` before trusting the number.
"""

from __future__ import annotations

import bisect
import importlib.util
import math
from pathlib import Path

from ..models import Evidence, Finding, SourceFile
from .base import Detector, register

MIN_TOKENS = 48


@register
class LanguageModel(Detector):
    name = "lm"
    title = "Языковая модель"
    description = "Binoculars: отношение перплексии к кросс-перплексии двух код-моделей (GPU, если есть)."
    weight = 1.0
    version = 1
    cost = 10
    options = {
        "observer": "Qwen/Qwen2.5-Coder-0.5B",
        "performer": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "device": "auto",
        "window": 512,
        "max_windows": 6,
        "batch": 2,
        "threshold": 0.85,
        "scale": 0.05,
    }

    def __init__(self, **overrides) -> None:
        super().__init__(**overrides)
        self._torch = None
        self._tokenizer = None
        self._observer = None
        self._performer = None
        self.device = "cpu"

    def unavailable_reason(self) -> str | None:
        # find_spec avoids the multi-second torch import when every file is already cached.
        missing = [name for name in ("torch", "transformers") if importlib.util.find_spec(name) is None]
        return f"не установлены: {', '.join(missing)}" if missing else None

    def prepare(self, root: Path) -> None:
        if self._performer is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        wanted = self.opt["device"]
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if wanted == "auto" else wanted
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(self.opt["performer"])
        self._observer = AutoModelForCausalLM.from_pretrained(self.opt["observer"], dtype=dtype).to(self.device).eval()
        self._performer = AutoModelForCausalLM.from_pretrained(self.opt["performer"], dtype=dtype).to(self.device).eval()

    def close(self) -> None:
        self._observer = self._performer = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _windows(self, ids: list[int]) -> list[tuple[int, int]]:
        size = self.opt["window"]
        spans = [(start, min(start + size, len(ids))) for start in range(0, len(ids), size)]
        if len(spans) > 1 and spans[-1][1] - spans[-1][0] < MIN_TOKENS:
            spans.pop()
        limit = self.opt["max_windows"]
        if len(spans) > limit:
            # Sample evenly across the file instead of only reading its head.
            spans = [spans[round(i * (len(spans) - 1) / (limit - 1))] for i in range(limit)]
        return spans

    def _score_batch(self, batch: list[list[int]]) -> list[tuple[float, float]]:
        """Return (log-perplexity, cross-perplexity) per window."""
        torch = self._torch
        width = max(len(ids) for ids in batch)
        input_ids = torch.zeros((len(batch), width), dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, ids in enumerate(batch):
            input_ids[row, : len(ids)] = torch.tensor(ids)
            mask[row, : len(ids)] = 1
        input_ids, mask = input_ids.to(self.device), mask.to(self.device)
        with torch.inference_mode():
            performer = self._performer(input_ids=input_ids, attention_mask=mask).logits[:, :-1].float()
            log_q = performer.log_softmax(-1)
            del performer
            targets = input_ids[:, 1:].unsqueeze(-1)
            nll = -log_q.gather(-1, targets).squeeze(-1)
            observer = self._observer(input_ids=input_ids, attention_mask=mask).logits[:, :-1].float().softmax(-1)
            cross = -(observer.mul_(log_q)).sum(-1)
            keep = mask[:, 1:].float()
            count = keep.sum(1)
            ppl = (nll * keep).sum(1) / count
            x_ppl = (cross * keep).sum(1) / count
        return list(zip(ppl.tolist(), x_ppl.tolist()))

    def _score_windows(self, windows: list[list[int]]) -> list[tuple[float, float]]:
        torch = self._torch
        size = self.opt["batch"]
        results: list[tuple[float, float]] = []
        for start in range(0, len(windows), size):
            batch = windows[start : start + size]
            try:
                results.extend(self._score_batch(batch))
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                for single in batch:
                    results.extend(self._score_batch([single]))
        return results

    def analyze(self, file: SourceFile) -> Finding | None:
        encoded = self._tokenizer(file.text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        if len(ids) < MIN_TOKENS:
            return None
        spans = self._windows(ids)
        scored = self._score_windows([ids[a:b] for a, b in spans])

        line_starts = [0]
        for line in file.text.splitlines(keepends=True):
            line_starts.append(line_starts[-1] + len(line))

        def line_of(token_index: int) -> int:
            return bisect.bisect_right(line_starts, offsets[token_index][0])

        threshold, scale = self.opt["threshold"], self.opt["scale"]

        def probability(ratio: float) -> float:
            return 1 / (1 + math.exp(-(threshold - ratio) / scale))

        tokens = total_ppl = total_x = 0.0
        chunks: list[Evidence] = []
        for (a, b), (ppl, x_ppl) in zip(spans, scored):
            n = b - a - 1
            tokens += n
            total_ppl += ppl * n
            total_x += x_ppl * n
            ratio = ppl / x_ppl if x_ppl > 0 else float("nan")
            chunks.append(Evidence(f"B={ratio:.3f}", line=line_of(a), end_line=line_of(b - 1), score=probability(ratio)))
        ratio = total_ppl / total_x
        chunks.sort(key=lambda e: -(e.score or 0))
        covered = sum(b - a for a, b in spans) / len(ids)
        # Short files give a noisy ratio; trust grows with the number of tokens seen.
        confidence = min(1.0, tokens / 400) * (0.5 + 0.5 * min(1.0, abs(ratio - threshold) / (2 * scale)))
        return Finding(
            score=probability(ratio),
            confidence=max(0.15, confidence),
            summary=f"Binoculars {ratio:.3f} (порог {threshold}; ниже — ближе к LLM)",
            evidence=chunks,
            metrics={
                "binoculars": round(ratio, 4),
                "log_ppl": round(total_ppl / tokens, 4),
                "cross_ppl": round(total_x / tokens, 4),
                "tokens": len(ids),
                "coverage": round(covered, 3),
            },
        )
