"""In-process bge-m3 embeddings (dense + sparse), via FlagEmbedding.

bge-m3 emits a dense vector and BM25-style sparse lexical weights in a single
forward pass - the hybrid signal the vector store indexes. It runs in-process
rather than through Ollama, which returns dense only (so it cannot feed hybrid
retrieval).

CPU is the default: ``use_fp16=False`` is bge-m3's documented CPU setting, and
the device is pinned to ``cpu`` rather than left to FlagEmbedding's auto-detect,
which would otherwise opportunistically pick MPS/CUDA and make results
machine-dependent. Both are overridable for a GPU host.

The model is heavy, so it loads lazily and is cached on first ``embed``: the
import, the weight download, and the load must not happen at module import or in
tests that inject a fake embedder.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from finrag.core.registry import registry
from finrag.core.types import EmbeddingVector

_BGE_M3_MODEL = "BAAI/bge-m3"


@lru_cache(maxsize=1)
def _load_model(model_name: str, use_fp16: bool, device: str) -> object:
    """Load and cache the bge-m3 model (lazy: imports FlagEmbedding on first use)."""
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(model_name, use_fp16=use_fp16, devices=device)


@registry.register("embedder", "bge_m3")
class BgeM3Embedder:
    """An :class:`~finrag.core.interfaces.Embedder` backed by bge-m3 (FlagEmbedding).

    Produces dense vectors and sparse lexical weights together, one
    :class:`~finrag.core.types.EmbeddingVector` per input text, preserving
    order. The ColBERT multi-vector output is not requested - there is no field
    for it and hybrid retrieval does not use it.
    """

    def __init__(
        self,
        *,
        model_name: str = _BGE_M3_MODEL,
        device: str = "cpu",
        use_fp16: bool = False,
        batch_size: int = 12,
        max_length: int = 8192,
    ) -> None:
        """Configure the embedder; the model itself loads lazily on first embed.

        Args:
          model_name: Hugging Face id of the bge-m3 model to load.
          device: Device to run on; ``"cpu"`` by default for deterministic,
            machine-independent output.
          use_fp16: Half precision - leave ``False`` on CPU (bge-m3's documented
            CPU setting); set ``True`` only for GPU inference.
          batch_size: Texts encoded per forward pass.
          max_length: Token cap per text; bge-m3's native ``8192`` so a
            context-prefixed chunk is never silently truncated.
        """
        self._model_name = model_name
        self._device = device
        self._use_fp16 = use_fp16
        self._batch_size = batch_size
        self._max_length = max_length

    def embed(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        """Return one dense+sparse embedding per text, in input order.

        Args:
          texts: The texts to embed.

        Returns:
          A list of :class:`EmbeddingVector`, aligned to ``texts``. Empty input
          yields an empty list without loading the model.
        """
        if not texts:
            return []
        model = _load_model(self._model_name, self._use_fp16, self._device)
        output = model.encode(  # type: ignore[attr-defined]
            list(texts),
            batch_size=self._batch_size,
            max_length=self._max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = output["dense_vecs"]
        sparse = output["lexical_weights"]
        return [
            EmbeddingVector(
                dense=[float(x) for x in dense[i]],
                sparse={int(token): float(weight) for token, weight in sparse[i].items()} or None,
            )
            for i in range(len(texts))
        ]
