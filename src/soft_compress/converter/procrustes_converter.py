import numpy as np
import torch

from base_converter import Converter, find_similar_tokens
from anchor_utils import get_vocab_file_anchor_embeddings


class ProcrustesConverter(Converter):
    """Orthogonal Procrustes alignment on vocab-file anchors (same as main-table LS)."""

    def __init__(
        self,
        src_model_path,
        tgt_model_path,
        common_vocab=None,
        converter_type="procrustes",
        max_anchors=None,
        **kwargs,
    ):
        if max_anchors is not None:
            kwargs["max_anchors"] = max_anchors
        super().__init__(src_model_path, tgt_model_path, common_vocab, converter_type, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        max_anchors = kwargs.get("max_anchors")

        if common_vocab is None:
            raise ValueError("ProcrustesConverter requires common_vocab (vocab_100k.txt)")

        cap = int(max_anchors) if max_anchors is not None else None
        self.src_embeddings, self.tgt_embeddings, self.n_anchors = get_vocab_file_anchor_embeddings(
            src_model_path, tgt_model_path, common_vocab, max_anchors=cap
        )
        label = "full vocab-file" if cap is None else f"cap={cap}"
        print(f"Procrustes anchors ({label}): {self.n_anchors}")

        self.projection_matrix = self._learn_projection()
        print(f"projection_matrix shape: {self.projection_matrix.shape}")

    def _learn_projection(self) -> torch.Tensor:
        x = self.src_embeddings.numpy().astype(np.float64)
        y = self.tgt_embeddings.numpy().astype(np.float64)
        cross = x.T @ y
        u, _, vt = np.linalg.svd(cross, full_matrices=False)
        w = u @ vt
        return torch.tensor(w, dtype=torch.float32).to(self.device)

    def convert(self, embedding):
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        embedding = embedding.to(self.device)
        return torch.matmul(embedding, self.projection_matrix)
