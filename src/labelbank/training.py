"""Contrastive training for the bi-encoder retriever.

Faithful to the competition loop: AdamW + OneCycleLR, gradient accumulation,
grad-norm clipping at 10.0, and the no-in-batch-negatives group loss at
temperature 0.01. Requires the ``[retrieve]`` extra.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .formatting import format_label, format_query
from .losses import no_in_batch_neg_loss
from .retriever import BiEncoderRetriever, encode_texts

__all__ = ["RetrieverTrainConfig", "train_retriever"]


@dataclass
class RetrieverTrainConfig:
    epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    lr: float = 1e-4
    weight_decay: float = 0.0
    temperature: float = 0.01
    one_cycle_pct_start: float = 0.1
    max_grad_norm: float = 10.0
    seed: int = 42
    shuffle: bool = True


def train_retriever(
    retriever: BiEncoderRetriever,
    queries: Sequence[str],
    candidate_pools: Sequence[Sequence[str]],
    cfg: Optional[RetrieverTrainConfig] = None,
) -> list:
    """Train the retriever on (query, [positive, hard negatives...]) groups.

    Args:
        retriever: A :class:`BiEncoderRetriever` whose model has trainable
            parameters (e.g. LoRA-wrapped via ``from_pretrained(trainable=True)``).
        queries: Raw query texts (prefixing is applied internally).
        candidate_pools: For each query, the label *texts* with the positive
            at index 0 — typically built by
            :func:`labelbank.mining.build_pools` + ``bank.texts_of``.
        cfg: Hyperparameters; defaults reproduce the competition recipe.

    Returns:
        The per-optimizer-step loss history.
    """
    import torch
    import torch.nn.functional as F
    from torch.nn.utils import clip_grad_norm_
    from torch.optim.lr_scheduler import OneCycleLR

    cfg = cfg or RetrieverTrainConfig()
    model, tokenizer = retriever.model, retriever.tokenizer

    pairs = [
        (
            format_query(q, retriever.query_prefix),
            [format_label(t, retriever.label_prefix) for t in pool],
        )
        for q, pool in zip(queries, candidate_pools)
    ]

    rng = random.Random(cfg.seed)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    steps_per_epoch = max(
        1, len(pairs) // cfg.batch_size // cfg.gradient_accumulation_steps
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=cfg.epochs * steps_per_epoch,
        pct_start=cfg.one_cycle_pct_start,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=100,
    )

    losses = []
    for _ in range(cfg.epochs):
        model.train()
        order = list(range(len(pairs)))
        if cfg.shuffle:
            rng.shuffle(order)

        for batch_idx in range(0, len(order), cfg.batch_size):
            batch = [pairs[i] for i in order[batch_idx : batch_idx + cfg.batch_size]]
            batch_queries = [q for q, _ in batch]
            batch_pools = [p for _, p in batch]

            q_emb = encode_texts(
                model, tokenizer, batch_queries, retriever.query_max_length, normalize=False
            )
            p_emb = encode_texts(
                model, tokenizer, batch_pools, retriever.label_max_length, normalize=False
            )
            q_emb = F.normalize(q_emb, p=2, dim=1)
            p_emb = F.normalize(p_emb, p=2, dim=1)

            _, loss = no_in_batch_neg_loss(q_emb, p_emb, temperature=cfg.temperature)
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)

            accum_step = batch_idx // cfg.batch_size + 1
            if accum_step % cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler.last_epoch < scheduler.total_steps - 1:
                    scheduler.step()
                losses.append(loss.item() * cfg.gradient_accumulation_steps)

    return losses
