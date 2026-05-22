import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from base_converter import Converter, find_similar_tokens

class LeastSquaresConverter(Converter):
    def __init__(self, src_model_path, tgt_model_path, common_vocab=None, converter_type="ls"):
        super().__init__(src_model_path, tgt_model_path, common_vocab, converter_type)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 获取源模型和目标模型的公共词汇表嵌入
        self.src_embeddings, self.tgt_embeddings = self._get_common_embeddings(common_vocab)

        # 学习线性映射矩阵
        self.projection_matrix = self._learn_projection()
        print(f"projection_matrix shape: {self.projection_matrix.shape}") 

    def _learn_projection(self):
        """使用最小二乘法学习投影矩阵"""
        X = self.src_embeddings.numpy()
        Y = self.tgt_embeddings.numpy()

        # 求解 XA = Y
        A, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        return torch.tensor(A, dtype=torch.float32).to(self.device)

    def convert(self, embedding):
        """
        转换输入嵌入向量

        Args:
            embedding: 输入嵌入向量 [batch_size, src_dim]
        Returns:
            转换后的嵌入向量 [batch_size, tgt_dim]
        """
        # start_time = time.time()
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        embedding = embedding.to(self.device)
        converted = torch.matmul(embedding, self.projection_matrix)

        return converted

if __name__ == "__main__":
    from pathlib import Path

    _here = Path(__file__).resolve().parent
    # 初始化转换器
    converter = LeastSquaresConverter(
        src_model_path="/data/Llama-3.2-1B-Instruct",
        tgt_model_path="/data/Llama-3.2-3B-Instruct",
        common_vocab=str(_here / "common_vocab_cased_be_ro_al.txt"),  # 可选
    )

    test_tokens = ["hello", "world", "AI"]
    find_similar_tokens(converter, test_tokens)