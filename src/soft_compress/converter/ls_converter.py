import torch
import numpy as np

from base_converter import Converter, find_similar_tokens
from anchor_utils import get_vocab_file_anchor_embeddings


class LeastSquaresConverter(Converter):
    def __init__(
        self,
        src_model_path,
        tgt_model_path,
        common_vocab=None,
        converter_type="ls",
        max_anchors=None,
        **kwargs,
    ):
        if max_anchors is not None:
            kwargs["max_anchors"] = max_anchors
        super().__init__(src_model_path, tgt_model_path, common_vocab, converter_type, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        max_anchors = kwargs.get("max_anchors")

        if common_vocab is not None:
            cap = int(max_anchors) if max_anchors is not None else None
            self.src_embeddings, self.tgt_embeddings, self.n_anchors = get_vocab_file_anchor_embeddings(
                src_model_path, tgt_model_path, common_vocab, max_anchors=cap
            )
            label = "full vocab-file" if cap is None else f"cap={cap}"
            print(f"LS anchors ({label}): {self.n_anchors}")
        else:
            self.src_embeddings, self.tgt_embeddings = self._get_common_embeddings(common_vocab)
            self.n_anchors = len(self.src_embeddings)

        self.projection_matrix = self._learn_projection()
        print(f"projection_matrix shape: {self.projection_matrix.shape}")

    def _learn_projection(self):
        """使用最小二乘法学习投影矩阵"""
        X = self.src_embeddings.numpy()
        Y = self.tgt_embeddings.numpy()

        A, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        return torch.tensor(A, dtype=torch.float32).to(self.device)

    def convert(self, embedding):
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        embedding = embedding.to(self.device)
        return torch.matmul(embedding, self.projection_matrix)


if __name__ == "__main__":
    from pathlib import Path

    _here = Path(__file__).resolve().parent
    converter = LeastSquaresConverter(
        src_model_path="/data/Llama-3.2-1B-Instruct",
        tgt_model_path="/data/Llama-3.2-3B-Instruct",
        common_vocab=str(_here / "common_vocab_cased_be_ro_al.txt"),
    )
    find_similar_tokens(converter, ["hello", "world", "AI"])
