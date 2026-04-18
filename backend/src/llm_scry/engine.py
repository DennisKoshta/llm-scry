from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Disable TransformerLens's beartype decorator for faster forwards.
os.environ.setdefault("TRANSFORMERS_LENS_ACCELERATE_TYPE_CHECKING", "0")

from transformer_lens import HookedTransformer  # noqa: E402

from .schemas import GenerateRequest, ModelInfo, TokenEvent, TopKAlternative
from .sessions import Session

logger = logging.getLogger(__name__)

# Activation patterns we want captured on the final forward pass.
# Using a function filter (not list) to catch all layers without enumerating.
_CAPTURE_PATTERNS = re.compile(
    r"blocks\.\d+\.(attn\.hook_pattern|hook_resid_pre|hook_resid_post|hook_attn_out|hook_mlp_out)$"
)


def _capture_filter(name: str) -> bool:
    return bool(_CAPTURE_PATTERNS.match(name))


@dataclass
class _StepResult:
    token_id: int
    logprob: float
    top_k: list[TopKAlternative]


class ModelEngine:
    def __init__(self) -> None:
        self.model: HookedTransformer | None = None
        self.name: str | None = None
        self.device: str = "cpu"

    def load(self, name: str, device: str | None = None) -> ModelInfo:
        target = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("loading model %s on %s", name, target)
        model = HookedTransformer.from_pretrained(name, device=target)
        model.eval()
        self.model = model
        self.name = name
        self.device = target
        return self.info()

    def info(self) -> ModelInfo:
        self._require_loaded()
        m = self.model
        assert m is not None
        cfg = m.cfg
        return ModelInfo(
            name=self.name or "",
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_model=cfg.d_model,
            d_vocab=cfg.d_vocab,
            device=self.device,
        )

    def _require_loaded(self) -> None:
        if self.model is None:
            raise RuntimeError("No model loaded. POST /model/load first.")

    def _decode(self, token_id: int) -> str:
        assert self.model is not None
        # to_string accepts a tensor; fall back to tokenizer for clarity.
        return self.model.tokenizer.decode([token_id])

    @torch.inference_mode()
    def _forward_step(
        self,
        tokens: torch.Tensor,
        top_k: int,
        temperature: float,
    ) -> _StepResult:
        assert self.model is not None
        logits = self.model(tokens)  # [1, seq, vocab]
        last = logits[0, -1, :]  # [vocab]
        log_probs = F.log_softmax(last, dim=-1)

        if temperature <= 0.0:
            chosen = int(torch.argmax(last).item())
        else:
            probs = F.softmax(last / temperature, dim=-1)
            chosen = int(torch.multinomial(probs, num_samples=1).item())

        chosen_logprob = float(log_probs[chosen].item())

        topk_vals, topk_idx = torch.topk(log_probs, k=top_k)
        alternatives = [
            TopKAlternative(
                token_id=int(idx.item()),
                token_str=self._decode(int(idx.item())),
                logprob=float(val.item()),
            )
            for val, idx in zip(topk_vals, topk_idx, strict=True)
        ]
        return _StepResult(token_id=chosen, logprob=chosen_logprob, top_k=alternatives)

    @torch.inference_mode()
    def _capture_cache(self, tokens: torch.Tensor) -> object:
        assert self.model is not None
        _, cache = self.model.run_with_cache(tokens, names_filter=_capture_filter)
        return cache

    def prepare_prompt(self, session: Session, prompt: str) -> torch.Tensor:
        """Tokenize prompt, populate session fields, return token tensor on device."""
        self._require_loaded()
        assert self.model is not None
        prompt_ids = self.model.to_tokens(prompt)  # [1, seq]
        session.prompt_token_ids = prompt_ids[0].tolist()
        session.prompt_token_strs = list(self.model.to_str_tokens(prompt))  # type: ignore[arg-type]
        return prompt_ids.to(self.device)

    async def stream_generate(
        self, session: Session, req: GenerateRequest, tokens: torch.Tensor
    ) -> AsyncIterator[TokenEvent]:
        self._require_loaded()
        assert self.model is not None

        for step in range(req.max_new_tokens):
            result = await asyncio.to_thread(
                self._forward_step, tokens, req.top_k, req.temperature
            )
            event = TokenEvent(
                position=len(session.prompt_token_strs) + step,
                token_id=result.token_id,
                token_str=self._decode(result.token_id),
                logprob=result.logprob,
                top_k=result.top_k,
            )
            session.tokens.append(event)
            yield event

            next_tok = torch.tensor([[result.token_id]], device=self.device)
            tokens = torch.cat([tokens, next_tok], dim=1)

            eos = self.model.tokenizer.eos_token_id
            if eos is not None and result.token_id == eos:
                break

        # Final capture pass so /session/{id}/* endpoints can serve activations.
        try:
            cache = await asyncio.to_thread(self._capture_cache, tokens)
            session.cache = cache  # type: ignore[assignment]
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("failed to capture activations: %s", e)


engine = ModelEngine()
