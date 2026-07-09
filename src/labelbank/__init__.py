"""labelbank — retrieve + rerank over a closed label bank.

Generalized from the silver-medal solution to Kaggle's "Eedi — Mining
Misconceptions in Mathematics": LLM bi-encoders trained with self-mined hard
negatives (no in-batch negatives), plus a generative listwise reranker.

Core install is torch-free (metrics, mining, formatting, data). Model-side
features live behind extras: ``pip install labelbank[retrieve]`` /
``[rerank]`` / ``[train]``.
"""

from .data import LabelBank, from_pairs, load_eedi
from .formatting import (
    DEFAULT_QUERY_PREFIX,
    RERANK_SYSTEM_PROMPT,
    build_rerank_text,
    format_label,
    format_query,
    render_chatml,
)
from .metrics import apk, mapk, recall_at_k
from .mining import build_pools, ensure_gold_in_top_k, gold_first_pool

__version__ = "0.2.1"

__all__ = [
    "LabelBank",
    "from_pairs",
    "load_eedi",
    "DEFAULT_QUERY_PREFIX",
    "RERANK_SYSTEM_PROMPT",
    "build_rerank_text",
    "format_label",
    "format_query",
    "render_chatml",
    "apk",
    "mapk",
    "recall_at_k",
    "build_pools",
    "ensure_gold_in_top_k",
    "gold_first_pool",
    "__version__",
]


def __getattr__(name):
    # Lazy access to torch-dependent symbols so the core install stays light.
    if name in ("BiEncoderRetriever", "encode_texts", "last_token_pool"):
        from . import retriever

        return getattr(retriever, name)
    if name in ("no_in_batch_neg_loss", "compute_no_in_batch_neg_loss"):
        from . import losses

        return getattr(losses, name)
    if name in ("ListwiseReranker", "build_training_rows"):
        from . import reranker

        return getattr(reranker, name)
    if name in ("RetrieverTrainConfig", "train_retriever"):
        from . import training

        return getattr(training, name)
    raise AttributeError(f"module 'labelbank' has no attribute {name!r}")
