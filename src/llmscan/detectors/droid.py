"""DroidDetect: a ModernBERT classifier fine-tuned to tell human from AI code.

Model: project-droid/DroidDetect-Base (Apache-2.0), trained on DroidCollection.
It is off by default: on the small sanity sample used while building llmscan it
flagged well-known human libraries as machine-written about as often as real
LLM output. Enable it with `--detectors git,lm,droid` and judge it on
your own data with `llmscan-eval`.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from ..models import Evidence, Finding, SourceFile
from .base import Detector, register

# The classifier saw whole snippets in training, so feed it definitions rather than arbitrary windows.
_DEFINITION = re.compile(
    r"(?m)^(?=(?:async def|def|class|func|function|fn|pub|public|private|protected|export|impl|static|template|module)\b)"
)
MIN_CHARS = 200


@register
class DroidDetect(Detector):
    name = "droid"
    title = "DroidDetect"
    description = "Обученный классификатор ИИ-кода (ModernBERT, project-droid). Экспериментальный, выключен по умолчанию."
    weight = 1.0
    cost = 8
    enabled_by_default = False
    options = {
        "model": "project-droid/DroidDetect-Base",
        "encoder": "answerdotai/ModernBERT-base",
        "device": "auto",
        "max_tokens": 510,
        "max_chunks": 8,
    }

    def __init__(self, **overrides) -> None:
        super().__init__(**overrides)
        self._torch = None
        self._tokenizer = None
        self._modules = None
        self.device = "cpu"

    def unavailable_reason(self) -> str | None:
        # find_spec avoids the multi-second torch import when every file is already cached.
        missing = [name for name in ("torch", "transformers") if importlib.util.find_spec(name) is None]
        return f"не установлены: {', '.join(missing)}" if missing else None

    def prepare(self, root: Path) -> None:
        if self._modules is not None:
            return
        import torch
        from huggingface_hub import hf_hub_download
        from torch import nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self._torch = torch
        wanted = self.opt["device"]
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if wanted == "auto" else wanted
        # weights_only keeps the pickle loader from executing anything shipped in the checkpoint.
        state = torch.load(hf_hub_download(self.opt["model"], "pytorch_model.bin"), map_location="cpu", weights_only=True)
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(self.opt["encoder"]))
        prefix = "text_encoder."
        encoder.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})

        def linear(name: str) -> nn.Linear:
            weight = state[f"{name}.weight"]
            layer = nn.Linear(weight.shape[1], weight.shape[0])
            layer.load_state_dict({"weight": weight, "bias": state[f"{name}.bias"]})
            return layer

        modules = [encoder, linear("text_projection"), linear("classifier")]
        if self.device.startswith("cuda"):
            modules = [m.half() for m in modules]
        self._modules = [m.to(self.device).eval() for m in modules]
        self._tokenizer = AutoTokenizer.from_pretrained(self.opt["model"])

    def close(self) -> None:
        self._modules = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _machine_probability(self, ids: list[int]) -> float:
        torch = self._torch
        encoder, projection, classifier = self._modules
        tokenizer = self._tokenizer
        tokens = torch.tensor([[tokenizer.cls_token_id, *ids, tokenizer.sep_token_id]], device=self.device)
        with torch.inference_mode():
            hidden = encoder(input_ids=tokens, attention_mask=torch.ones_like(tokens)).last_hidden_state.mean(1)
            probabilities = classifier(torch.relu(projection(hidden))).float().softmax(-1)[0]
        # Class 0 is HUMAN_GENERATED; the rest are flavours of machine involvement.
        return 1 - probabilities[0].item()

    def analyze(self, file: SourceFile) -> Finding | None:
        chunks: list[tuple[int, str]] = []
        position = 0
        for part in _DEFINITION.split(file.text):
            if len(part.strip()) >= MIN_CHARS:
                chunks.append((file.text.count("\n", 0, position) + 1, part))
            position += len(part)
        if not chunks:
            if len(file.text.strip()) < MIN_CHARS:
                return None
            chunks = [(1, file.text)]
        limit = self.opt["max_chunks"]
        if len(chunks) > limit:
            chunks = [chunks[round(i * (len(chunks) - 1) / (limit - 1))] for i in range(limit)]

        weighted = total = 0.0
        evidence: list[Evidence] = []
        for line, text in chunks:
            ids = self._tokenizer(text, add_special_tokens=False)["input_ids"][: self.opt["max_tokens"]]
            probability = self._machine_probability(ids)
            weighted += probability * len(ids)
            total += len(ids)
            evidence.append(Evidence("фрагмент", line=line, end_line=line + text.count("\n"), score=probability))
        evidence.sort(key=lambda e: -(e.score or 0))
        score = weighted / total
        return Finding(
            score=score,
            confidence=min(1.0, total / 400) * (0.3 + 0.7 * abs(score - 0.5) * 2),
            summary=f"P(машина) = {score:.2f} по {len(chunks)} фрагментам",
            evidence=evidence,
            metrics={"p_machine": round(score, 4), "chunks": len(chunks)},
        )
