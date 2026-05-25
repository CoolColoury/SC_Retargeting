import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from anchor_utils import get_vocab_file_anchor_embeddings
from base_converter import Converter, find_similar_tokens


class BPConverter(Converter):
    """Anchor-relative BP search converter (appendix negative baseline)."""

    def __init__(
        self,
        src_model_path,
        tgt_model_path,
        common_vocab=None,
        converter_type="bp",
        max_anchors=None,
        lr=1e-2,
        num_iterations=500,
        lambd=0.99,
        verbose=False,
        **kwargs,
    ):
        if max_anchors is not None:
            kwargs["max_anchors"] = max_anchors
        super().__init__(src_model_path, tgt_model_path, common_vocab, converter_type, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.num_iterations = int(num_iterations)
        self.lambd = lambd
        self.verbose = bool(verbose)
        max_anchors = kwargs.get("max_anchors")

        if common_vocab is not None:
            cap = int(max_anchors) if max_anchors is not None else None
            self.src_anchor_embeddings, self.tgt_anchor_embeddings, self.n_anchors = (
                get_vocab_file_anchor_embeddings(
                    src_model_path, tgt_model_path, common_vocab, max_anchors=cap
                )
            )
            label = "full vocab-file" if cap is None else f"cap={cap}"
            print(f"BP anchors ({label}): {self.n_anchors}")
        else:
            self.src_anchor_embeddings, self.tgt_anchor_embeddings = self._get_common_embeddings(
                common_vocab
            )
            self.n_anchors = len(self.src_anchor_embeddings)

        self.src_embeddings = self._get_embeddings(src_model_path).cpu()
        self.tgt_embeddings = self._get_embeddings(tgt_model_path).cpu()
        self.reset_timing_stats()

    def reset_timing_stats(self) -> None:
        self._search_seconds = 0.0
        self._search_calls = 0

    def get_timing_stats(self) -> dict:
        calls = self._search_calls
        seconds = self._search_seconds
        return {
            "bp_search_seconds": float(seconds),
            "bp_search_calls": int(calls),
            "bp_sec_per_search": float(seconds / calls) if calls else 0.0,
            "bp_num_iterations": self.num_iterations,
            "bp_n_anchors": int(self.n_anchors),
        }

    def _embedding2relative(self, embedding, anchor_embeddings):
        A = embedding.to(torch.float32).to(self.device)
        B = anchor_embeddings.to(torch.float32).to(self.device)

        if A.dim() == 1:
            A = A.unsqueeze(0)

        direction_similarity = F.normalize(A, dim=-1) @ F.normalize(B, dim=-1).t()
        A_norm = torch.norm(A, dim=-1, keepdim=True)
        B_norm = torch.norm(B, dim=-1, keepdim=True).t()
        magnitude_encoding = torch.log(A_norm + 1e-12) - torch.log(B_norm + 1e-12)

        return {
            "direction_similarity": direction_similarity,
            "magnitude_encoding": magnitude_encoding,
        }

    def _initialization(self, shape, tgt_anchor_embeddings):
        candidate_embeddings = torch.empty(shape, dtype=torch.float32).to(self.device)
        tgt_mean = torch.mean(tgt_anchor_embeddings, dim=0)
        tgt_std = torch.std(tgt_anchor_embeddings, dim=0)
        mean_val = float(torch.mean(tgt_mean))
        std_val = float(torch.mean(tgt_std))
        with torch.no_grad():
            candidate_embeddings.normal_(mean=mean_val, std=std_val)
        return candidate_embeddings

    def _search(self, embedding: torch.Tensor) -> torch.Tensor:
        t0 = time.perf_counter()
        embedding = embedding.to(torch.float32).to(self.device)

        with torch.enable_grad():
            src_rels = self._embedding2relative(embedding, self.src_anchor_embeddings)
            candidate_embeddings = self._initialization(
                (embedding.shape[0], self.tgt_anchor_embeddings.shape[1]),
                self.tgt_anchor_embeddings,
            )
            candidate_embeddings = candidate_embeddings.clone().detach().requires_grad_(True)
            optimizer = torch.optim.AdamW([candidate_embeddings], lr=self.lr)
            best_loss = 1e9
            best_candidate_embeddings = candidate_embeddings

            for i in range(self.num_iterations):
                x_rel = self._embedding2relative(candidate_embeddings, self.tgt_anchor_embeddings)
                direction_loss = 1 - F.cosine_similarity(
                    x_rel["direction_similarity"],
                    src_rels["direction_similarity"],
                    dim=-1,
                ).mean()
                magnitude_loss = F.smooth_l1_loss(
                    x_rel["magnitude_encoding"],
                    src_rels["magnitude_encoding"],
                )
                loss = self.lambd * direction_loss + (1 - self.lambd) * magnitude_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_candidate_embeddings = candidate_embeddings.detach().clone()

                if self.verbose and i % 50 == 0:
                    print(
                        f"  BP iter {i}: loss={loss.item():.6f}, "
                        f"dir={direction_loss.item():.6f}, mag={magnitude_loss.item():.6f}"
                    )

        self._search_seconds += time.perf_counter() - t0
        self._search_calls += 1
        return best_candidate_embeddings

    def convert(self, embedding):
        if embedding.dim() == 1:
            return self._search(embedding.unsqueeze(0)).squeeze(0)

        if embedding.dim() == 2:
            outputs = [self._search(vec.unsqueeze(0)).squeeze(0) for vec in embedding]
            return torch.stack(outputs)

        if embedding.dim() == 3:
            batch_outputs = []
            for batch_idx in range(embedding.shape[0]):
                mem_outputs = [
                    self._search(embedding[batch_idx, mem_idx].unsqueeze(0)).squeeze(0)
                    for mem_idx in range(embedding.shape[1])
                ]
                batch_outputs.append(torch.stack(mem_outputs))
            return torch.stack(batch_outputs)

        raise ValueError(f"Unsupported embedding shape: {tuple(embedding.shape)}")


if __name__ == "__main__":
    converter = BPConverter(
        src_model_path="/data/Llama-3.2-1B-Instruct",
        tgt_model_path="/data/Llama-3.2-3B-Instruct",
    )
    find_similar_tokens(converter, ["hello", "world", "AI"])
