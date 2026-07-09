"""Generative listwise reranker over the retriever's top-k candidates.

Candidates are inlined into the prompt as lettered options (A, B, ...); the
model is fine-tuned (completion-only) to emit the gold letter, with the gold
position shuffled at training time so no position prior is learned. At
inference, the next-token logits over the option letters re-order the
candidates. Requires the ``[rerank]`` extra.

The library path uses plain ``transformers`` + ``trl``; the original
competition run used Unsloth for speed — preserved verbatim under
``competition/``.
"""

from __future__ import annotations

import random
import string
from typing import Optional, Sequence

from .formatting import RERANK_SYSTEM_PROMPT, build_rerank_text, render_chatml
from .mining import ensure_gold_in_top_k

__all__ = ["build_training_rows", "ListwiseReranker"]


def build_training_rows(
    query_texts: Sequence[str],
    ranked_candidate_texts: Sequence[Sequence[str]],
    gold_texts: Sequence[str],
    k: int = 5,
    candidate_noun: str = "misconception",
    system: str = RERANK_SYSTEM_PROMPT,
    seed: int = 42,
) -> list:
    """Build ChatML training strings from retriever rankings.

    For each query: take the top-k candidate texts, force the gold in if the
    retriever missed it, shuffle, and render the ChatML example split at the
    assistant tag, so the SFT loss (``completion_only_loss``) is computed
    only over the gold letter.

    Returns:
        List of dicts with ``prompt`` (everything through
        ``<|im_start|>assistant\n``), ``completion`` (the gold letter +
        closing tag), ``letter`` (gold letter) and ``gold_index``.
    """
    rng = random.Random(seed)
    rows = []
    for query, ranked, gold in zip(query_texts, ranked_candidate_texts, gold_texts):
        candidates, gold_idx = ensure_gold_in_top_k(list(ranked), gold, k=k, rng=rng)
        letter = string.ascii_uppercase[gold_idx]
        user_text = build_rerank_text(query, candidates, candidate_noun=candidate_noun)
        full = render_chatml(user_text, answer_letter=letter, system=system)
        prompt = render_chatml(user_text, answer_letter="", system=system)
        rows.append(
            {
                "prompt": prompt,
                "completion": full[len(prompt):],
                "letter": letter,
                "gold_index": gold_idx,
            }
        )
    return rows


class ListwiseReranker:
    """Score retriever candidates with a generative model's letter logits."""

    def __init__(
        self,
        model,
        tokenizer,
        candidate_noun: str = "misconception",
        system: str = RERANK_SYSTEM_PROMPT,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.candidate_noun = candidate_noun
        self.system = system

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        **kwargs,
    ) -> "ListwiseReranker":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ._compat import model_dtype_kwargs

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True,
            **model_dtype_kwargs(getattr(torch, dtype)),
        )
        return cls(model, tokenizer, **kwargs)

    def _letter_token_ids(self, n: int) -> list:
        ids = []
        for letter in string.ascii_uppercase[:n]:
            toks = self.tokenizer.encode(letter, add_special_tokens=False)
            ids.append(toks[0])
        return ids

    def rerank(
        self,
        query_text: str,
        candidate_texts: Sequence[str],
    ) -> list:
        """Return candidate indices reordered by the model's letter logits."""
        import torch

        user_text = build_rerank_text(
            query_text, candidate_texts, candidate_noun=self.candidate_noun
        )
        prompt = render_chatml(user_text, answer_letter="", system=self.system)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]

        letter_ids = self._letter_token_ids(len(candidate_texts))
        letter_scores = logits[letter_ids]
        order = torch.argsort(letter_scores, descending=True).tolist()
        return order

    def train(
        self,
        rows: Sequence[dict],
        output_dir: str,
        lora: Optional[dict] = None,
        max_seq_length: int = 1024,
        epochs: int = 1,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        lr: float = 1e-4,
        seed: int = 42,
    ):
        """Completion-only SFT on rows from :func:`build_training_rows`.

        Uses a ``prompt``/``completion`` dataset with ``completion_only_loss``
        instead of ``DataCollatorForCompletionOnlyLM`` + a response-template
        string match: current trl releases removed that collator, and the
        dataset-shape-driven path also sidesteps any brittleness from the
        tokenizer re-merging the ``<|im_start|>assistant\\n`` marker.
        """
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from trl import SFTConfig, SFTTrainer

        if lora is not None:
            lora_config = LoraConfig(
                r=lora.get("r", 16),
                lora_alpha=lora.get("alpha", 32),
                lora_dropout=lora.get("dropout", 0.0),
                bias=lora.get("bias", "none"),
                task_type="CAUSAL_LM",
                target_modules=lora.get(
                    "target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                ),
            )
            self.model = get_peft_model(self.model, lora_config)

        dataset = Dataset.from_list(
            [{"prompt": r["prompt"], "completion": r["completion"]} for r in rows]
        )

        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                output_dir=output_dir,
                max_length=max_seq_length,
                completion_only_loss=True,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=lr,
                lr_scheduler_type="linear",
                weight_decay=0.01,
                logging_steps=5,
                seed=seed,
                report_to="none",
            ),
        )
        trainer.train()
        return trainer
