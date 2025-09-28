"""Utilities for retrieving similar words with FAISS."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np


Params = Tuple[Dict[str, int], Dict[int, str], np.ndarray]


def load_params(path: str) -> Params:
    """Load pretrained parameters from ``path``.

    The expected pickle file contains ``word_to_id``, ``id_to_word`` and
    ``word_vec`` entries.
    """
    with open(path, "rb") as file:
        params = pickle.load(file)

    word_to_id: Dict[str, int] = params["word_to_id"]
    id_to_word: Dict[int, str] = params["id_to_word"]
    word_vec: np.ndarray = params["word_vec"].astype(np.float32)
    return word_to_id, id_to_word, word_vec


def l2_normalize(vectors: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    """Safely normalise ``vectors`` along ``axis`` using the L2 norm."""
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    norms = np.maximum(norms, eps)
    return vectors / norms


@dataclass
class WordEmbeddingSearcher:
    """Wrapper around FAISS for similarity search on word vectors."""

    word_to_id: Dict[str, int]
    id_to_word: Dict[int, str]
    embeddings: np.ndarray
    normalized_embeddings: np.ndarray
    index: faiss.Index

    @classmethod
    def from_pickle(cls, path: str) -> "WordEmbeddingSearcher":
        """Create a searcher from a pickled parameter file."""
        word_to_id, id_to_word, embeddings = load_params(path)
        normalized_embeddings = l2_normalize(embeddings, axis=1)

        dim = normalized_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(normalized_embeddings)

        return cls(word_to_id, id_to_word, embeddings, normalized_embeddings, index)

    def context_vector(
        self,
        context_ids: Sequence[int],
        weights: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Return a normalised context vector for the provided word IDs."""
        context_matrix = self.embeddings[np.asarray(context_ids, dtype=np.int32)]

        if weights is None:
            context = context_matrix.mean(axis=0)
        else:
            weights_array = np.asarray(weights, dtype=np.float32)
            weight_sum = weights_array.sum()
            if np.isclose(weight_sum, 0.0):
                context = context_matrix.mean(axis=0)
            else:
                weights_array = weights_array / weight_sum
                context = np.average(context_matrix, axis=0, weights=weights_array)

        context = context.astype(np.float32, copy=False)
        return l2_normalize(context[None, :], axis=1)[0]

    def predict_topk(
        self,
        context_ids: Sequence[int],
        k: int = 5,
        banned_ids: Optional[Iterable[int]] = None,
        weights: Optional[Sequence[float]] = None,
    ) -> List[Tuple[str, float]]:
        """Return the ``k`` nearest neighbours for a CBOW-style context."""
        query_vector = self.context_vector(context_ids, weights)[None, :]

        banned_set = set(context_ids)
        if banned_ids is not None:
            banned_set.update(banned_ids)

        search_k = max(k + len(banned_set), k)
        similarities, indices = self.index.search(query_vector, search_k)

        similarities_list = similarities[0].tolist()
        indices_list = indices[0].tolist()

        filtered_pairs = [
            (idx, sim)
            for idx, sim in zip(indices_list, similarities_list)
            if idx not in banned_set
        ]

        top_results = filtered_pairs[:k]
        return [
            (self.id_to_word[candidate_id], float(similarity))
            for candidate_id, similarity in top_results
        ]


def _example() -> None:
    """Tiny example showing how to load a model and query the index."""
    searcher = WordEmbeddingSearcher.from_pickle("ptb_params.pkl")

    context_words = ["you", "toyota", "we"]
    context_ids = [searcher.word_to_id[word] for word in context_words]

    predictions = searcher.predict_topk(context_ids, k=5, banned_ids=context_ids)
    print(predictions)


if __name__ == "__main__":
    _example()
