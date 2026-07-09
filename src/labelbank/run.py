"""Config-driven pipelines: the full retrieve → mine → retrain loop, and the
reranker stage, from one YAML file.

Retriever stage (``stage: retriever``)::

    python -m labelbank.run --cfg examples/configs/quickstart.yaml

loads (query, gold_id) pairs and the label bank, evaluates the zero-shot
backbone, trains one *bootstrap* round on random negatives (mining from the
untrained embedder collapses retrieval quality — see the README ablation),
then runs ``mining_rounds`` rounds of: rank the whole bank for every training
query → build gold-first hard-negative pools from the previous round →
train with the no-in-batch-negatives loss → evaluate. Saves the adapter,
per-split rankings (``rankings.parquet``) and ``metrics.json`` under
``output_dir``.

Rounds continue training the same adapter. The competition ran each round as
a separate script invocation with a fresh adapter; for that protocol see
``examples/mined_negatives_experiment.py``, which measures both.

Reranker stage (``stage: reranker``) consumes the retriever's
``rankings.parquet`` and trains the generative listwise reranker
(completion-only SFT, gold position shuffled).

Requires the ``[train]`` extra.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import pandas as pd
import yaml

from .data import LabelBank, load_eedi
from .mining import build_pools

__all__ = ["RunConfig", "load_config", "run_retriever", "run_reranker"]

_STAGES = ("retriever", "reranker")
_FORMATS = ("generic", "eedi")


@dataclass
class RunConfig:
    """Everything needed for one pipeline stage. Loadable via :func:`load_config`."""

    stage: str = "retriever"
    output_dir: str = "./output/labelbank"
    seed: int = 42

    # data
    data_format: str = "generic"  # "generic": csv/parquet with query/gold columns
    train_path: str = ""          # generic file, or the Eedi train csv
    bank_path: str = ""           # generic bank csv, or misconception_mapping.csv
    query_col: str = "query"
    gold_col: str = "gold_id"
    id_col: str = "id"
    text_col: str = "text"
    eval_holdout: float = 0.05    # random holdout when no fold split is used
    fold_col: str = "fold"
    eval_fold: Optional[int] = None  # use this fold as eval (needs fold_col)

    # model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    query_prefix: str = ""
    label_prefix: str = ""
    query_max_length: int = 512
    label_max_length: int = 64
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0

    # retriever stage
    mining_rounds: int = 1
    pool_size: int = 25
    epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    lr: float = 1e-4
    temperature: float = 0.01
    weight_decay: float = 0.0
    eval_batch_size: int = 8
    eval_ks: List[int] = field(default_factory=lambda: [1, 10, 25, 50, 100])
    map_k: int = 25

    # reranker stage
    rankings_path: str = ""       # rankings.parquet from the retriever stage
    rerank_k: int = 5
    candidate_noun: str = "label"
    rerank_epochs: int = 1
    rerank_batch_size: int = 2
    rerank_gradient_accumulation_steps: int = 4
    rerank_lr: float = 1e-4
    max_seq_length: int = 1024

    def __post_init__(self) -> None:
        if self.stage not in _STAGES:
            raise ValueError(f"stage must be one of {_STAGES}, got {self.stage!r}")
        if self.data_format not in _FORMATS:
            raise ValueError(
                f"data_format must be one of {_FORMATS}, got {self.data_format!r}"
            )


def load_config(path: str) -> RunConfig:
    """Load a YAML file into a :class:`RunConfig`, rejecting unknown keys."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    known = set(RunConfig.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    return RunConfig(**raw)


def _read_table(path: str) -> pd.DataFrame:
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_bank(cfg: RunConfig) -> LabelBank:
    if cfg.data_format == "eedi":
        mapping = pd.read_csv(cfg.bank_path)
        return LabelBank(
            ids=mapping["MisconceptionId"].tolist(),
            texts=mapping["MisconceptionName"].tolist(),
        )
    return LabelBank.from_csv(cfg.bank_path, id_col=cfg.id_col, text_col=cfg.text_col)


def load_data(cfg: RunConfig):
    """Load (train_df, eval_df, bank); frames have ``query`` / ``gold_id``."""
    bank = _load_bank(cfg)

    if cfg.data_format == "eedi":
        df, _ = load_eedi(cfg.train_path, cfg.bank_path)
        df = df.rename(columns={"AllText": "query", "MisconceptionId": "gold_id"})
    else:
        df = _read_table(cfg.train_path).rename(
            columns={cfg.query_col: "query", cfg.gold_col: "gold_id"}
        )

    if cfg.eval_fold is not None and cfg.fold_col in df.columns:
        eval_df = df[df[cfg.fold_col] == cfg.eval_fold].reset_index(drop=True)
        train_df = df[df[cfg.fold_col] != cfg.eval_fold].reset_index(drop=True)
    else:
        eval_df = df.sample(frac=cfg.eval_holdout, random_state=cfg.seed)
        train_df = df.drop(eval_df.index).reset_index(drop=True)
        eval_df = eval_df.reset_index(drop=True)

    return train_df, eval_df, bank


def _build_retriever(cfg: RunConfig, trainable: bool):
    from .retriever import BiEncoderRetriever

    return BiEncoderRetriever.from_pretrained(
        cfg.model_name,
        trainable=trainable,
        lora={"r": cfg.lora_r, "alpha": cfg.lora_alpha, "dropout": cfg.lora_dropout},
        load_in_4bit=cfg.load_in_4bit,
        dtype=cfg.dtype,
        query_prefix=cfg.query_prefix,
        label_prefix=cfg.label_prefix,
        query_max_length=cfg.query_max_length,
        label_max_length=cfg.label_max_length,
    )


def _evaluate(retriever, df, bank, cfg: RunConfig) -> dict:
    result = retriever.evaluate(
        df["query"].tolist(),
        df["gold_id"].tolist(),
        bank,
        ks=tuple(cfg.eval_ks),
        map_k=cfg.map_k,
        batch_size=cfg.eval_batch_size,
    )
    result.pop("ranked")
    return {k: round(float(v), 4) for k, v in result.items()}


def run_retriever(cfg: RunConfig) -> dict:
    """The full retrieve → mine → retrain loop; returns the metrics dict."""
    from .training import RetrieverTrainConfig, train_retriever

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "run_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    train_df, eval_df, bank = load_data(cfg)
    retriever = _build_retriever(cfg, trainable=True)

    metrics = {"zero_shot": _evaluate(retriever, eval_df, bank, cfg)}
    print(f"zero_shot: {metrics['zero_shot']}")

    train_queries = train_df["query"].tolist()
    train_golds = train_df["gold_id"].tolist()

    # Bootstrap round: one epoch on *random* negatives before the first
    # mining round. Mining round 1 straight from the zero-shot model's
    # rankings collapses retrieval quality (MAP 0.43 vs 0.80 for random
    # negatives on banking77 — see examples/mined_negatives_experiment.py
    # --cold-start) because an untrained embedder's "hardest" negatives are
    # mostly noise, not genuine near-misses. Mirrors that script's arm 2.
    rng = random.Random(cfg.seed)
    bootstrap_pools = [
        [gold] + rng.sample([i for i in bank.ids if i != gold], cfg.pool_size - 1)
        for gold in train_golds
    ]
    train_retriever(
        retriever,
        train_queries,
        [bank.texts_of(p) for p in bootstrap_pools],
        RetrieverTrainConfig(
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            lr=cfg.lr,
            temperature=cfg.temperature,
            weight_decay=cfg.weight_decay,
            seed=cfg.seed,
        ),
    )
    metrics["bootstrap"] = _evaluate(retriever, eval_df, bank, cfg)
    print(f"bootstrap: {metrics['bootstrap']}")

    train_rankings = retriever.retrieve(
        train_queries, bank, batch_size=cfg.eval_batch_size
    )

    for round_idx in range(1, cfg.mining_rounds + 1):
        pools = build_pools(train_rankings, train_golds, top_k=cfg.pool_size)
        train_retriever(
            retriever,
            train_queries,
            [bank.texts_of(p) for p in pools],
            RetrieverTrainConfig(
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                lr=cfg.lr,
                temperature=cfg.temperature,
                weight_decay=cfg.weight_decay,
                seed=cfg.seed + round_idx,
            ),
        )
        metrics[f"round_{round_idx}"] = _evaluate(retriever, eval_df, bank, cfg)
        print(f"round_{round_idx}: {metrics[f'round_{round_idx}']}")
        train_rankings = retriever.retrieve(
            train_queries, bank, batch_size=cfg.eval_batch_size
        )

    # Save adapter (PEFT) or full model, the rankings both stages need, metrics.
    save_dir = os.path.join(cfg.output_dir, "adapter")
    retriever.model.save_pretrained(save_dir)
    retriever.tokenizer.save_pretrained(save_dir)

    eval_rankings = retriever.retrieve(
        eval_df["query"].tolist(), bank, batch_size=cfg.eval_batch_size
    )
    rankings = pd.DataFrame(
        {
            "split": ["train"] * len(train_df) + ["eval"] * len(eval_df),
            "query": train_queries + eval_df["query"].tolist(),
            "gold_id": train_golds + eval_df["gold_id"].tolist(),
            "ranked_ids": train_rankings + eval_rankings,
        }
    )
    rankings.to_parquet(os.path.join(cfg.output_dir, "rankings.parquet"))

    with open(os.path.join(cfg.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def run_reranker(cfg: RunConfig):
    """Train the listwise reranker on the retriever stage's rankings."""
    from .reranker import ListwiseReranker, build_training_rows

    os.makedirs(cfg.output_dir, exist_ok=True)
    bank = _load_bank(cfg)
    rankings = pd.read_parquet(cfg.rankings_path)
    train_rows = rankings[rankings["split"] == "train"]

    rows = build_training_rows(
        train_rows["query"].tolist(),
        [bank.texts_of(list(r)[: cfg.rerank_k]) for r in train_rows["ranked_ids"]],
        bank.texts_of(train_rows["gold_id"].tolist()),
        k=cfg.rerank_k,
        candidate_noun=cfg.candidate_noun,
        seed=cfg.seed,
    )

    reranker = ListwiseReranker.from_pretrained(
        cfg.model_name, dtype=cfg.dtype, candidate_noun=cfg.candidate_noun
    )
    trainer = reranker.train(
        rows,
        output_dir=os.path.join(cfg.output_dir, "reranker"),
        lora={"r": cfg.lora_r, "alpha": cfg.lora_alpha, "dropout": cfg.lora_dropout},
        max_seq_length=cfg.max_seq_length,
        epochs=cfg.rerank_epochs,
        batch_size=cfg.rerank_batch_size,
        gradient_accumulation_steps=cfg.rerank_gradient_accumulation_steps,
        lr=cfg.rerank_lr,
        seed=cfg.seed,
    )
    return trainer


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, help="Path to a YAML config")
    args = parser.parse_args(argv)
    cfg = load_config(args.cfg)
    if cfg.stage == "retriever":
        metrics = run_retriever(cfg)
        print(json.dumps(metrics, indent=2))
    else:
        run_reranker(cfg)


if __name__ == "__main__":
    main()
