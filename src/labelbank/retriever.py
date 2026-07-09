"""LLM bi-encoder retrieval over a label bank.

Any Hugging Face causal/encoder backbone becomes an embedder via last-token
pooling; LoRA and 4-bit quantization make multi-billion-parameter backbones
trainable on a small number of GPUs. Requires the ``[retrieve]`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from .data import LabelBank
from .formatting import format_label, format_query
from .metrics import mapk, recall_at_k

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["last_token_pool", "encode_texts", "BiEncoderRetriever"]


def last_token_pool(last_hidden_states, attention_mask):
    """Pool the hidden state of each sequence's final non-padding token.

    Handles both left- and right-padded batches (left padding is detected
    when every sequence's last position is unmasked).
    """
    import torch

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def encode_texts(model, tokenizer, texts, max_length: int, normalize: bool = True):
    """Tokenize, forward, last-token-pool (and optionally L2-normalize).

    Accepts a flat list of strings or a list of per-query groups (which is
    flattened, preserving group order) — matching the training-loop contract
    where each query's passage group is ``[positive, negatives...]``.
    """
    import torch
    import torch.nn.functional as F

    if isinstance(texts[0], list):
        texts = [text for group in texts for text in group]

    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    )
    input_ids = encodings["input_ids"].to(model.device)
    attention_mask = encodings["attention_mask"].to(model.device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


class BiEncoderRetriever:
    """A bi-encoder over a label bank, backed by any HF model.

    Args:
        model: A loaded HF model exposing ``last_hidden_state`` (e.g. from
            ``AutoModel.from_pretrained``), optionally PEFT-wrapped.
        tokenizer: The matching tokenizer.
        query_prefix: Instruction prepended to every query (see
            :data:`labelbank.formatting.DEFAULT_QUERY_PREFIX` for the
            competition wording).
        label_prefix: Prefix for bank entries (competition used none).
        query_max_length / label_max_length: Truncation budgets.
    """

    def __init__(
        self,
        model,
        tokenizer,
        query_prefix: str = "",
        label_prefix: str = "",
        query_max_length: int = 512,
        label_max_length: int = 64,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.query_prefix = query_prefix
        self.label_prefix = label_prefix
        self.query_max_length = query_max_length
        self.label_max_length = label_max_length

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        lora: Optional[dict] = None,
        load_in_4bit: bool = False,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        trainable: bool = False,
        **kwargs,
    ) -> "BiEncoderRetriever":
        """Load a backbone (optionally 4-bit quantized, optionally LoRA-wrapped)."""
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch_dtype = getattr(torch, dtype)
        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )

        from ._compat import model_dtype_kwargs

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True,
            quantization_config=quantization_config,
            **model_dtype_kwargs(torch_dtype),
        )
        model.config.use_cache = False

        if trainable:
            from peft import LoraConfig, TaskType, get_peft_model

            lora = lora or {}
            lora_config = LoraConfig(
                r=lora.get("r", 16),
                lora_alpha=lora.get("alpha", 32),
                lora_dropout=lora.get("dropout", 0.0),
                bias=lora.get("bias", "none"),
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=lora.get(
                    "target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                ),
            )
            model = get_peft_model(model, lora_config)

        return cls(model, tokenizer, **kwargs)

    def _embed(self, texts: Sequence[str], max_length: int, batch_size: int):
        import torch

        chunks = []
        self.model.eval()
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            with torch.no_grad():
                emb = encode_texts(self.model, self.tokenizer, batch, max_length)
            chunks.append(emb)
        return torch.cat(chunks, dim=0)

    def embed_queries(self, queries: Sequence[str], batch_size: int = 4):
        formatted = [format_query(q, self.query_prefix) for q in queries]
        return self._embed(formatted, self.query_max_length, batch_size)

    def embed_labels(self, texts: Sequence[str], batch_size: int = 4):
        formatted = [format_label(t, self.label_prefix) for t in texts]
        return self._embed(formatted, self.label_max_length, batch_size)

    def retrieve(
        self,
        queries: Sequence[str],
        bank: LabelBank,
        top_k: Optional[int] = None,
        batch_size: int = 4,
    ) -> list:
        """Rank bank entries for each query.

        Returns a list of ranked label-id lists (full ranking, or ``top_k``).
        """
        q = self.embed_queries(queries, batch_size)
        p = self.embed_labels(bank.texts, batch_size)
        scores = (q @ p.T).float().cpu().numpy()
        order = np.argsort(-scores, axis=1)
        if top_k is not None:
            order = order[:, :top_k]
        ids = np.asarray(bank.ids)
        return [ids[row].tolist() for row in order]

    def evaluate(
        self,
        queries: Sequence[str],
        gold_ids: Sequence,
        bank: LabelBank,
        ks: Sequence[int] = (1, 10, 25, 50, 100),
        map_k: int = 25,
        batch_size: int = 4,
    ) -> dict:
        """MAP@K plus a Recall@K ladder against the full bank."""
        ranked = self.retrieve(queries, bank, top_k=None, batch_size=batch_size)
        result = {f"map@{map_k}": mapk([[g] for g in gold_ids], [r[:map_k] for r in ranked], k=map_k)}
        result.update(recall_at_k(gold_ids, ranked, ks))
        result["ranked"] = ranked
        return result
