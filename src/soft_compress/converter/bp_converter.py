import torch
import torch.nn.functional as F

from base_converter import Converter, find_similar_tokens

from transformers import AutoTokenizer

DEBUG = True

class BPConverter(Converter):
    """
    BPConverter: 使用相对表示空间进行 embedding 转换
    
    使用 Cosine Loss：
    - Direction: 1 - cosine_similarity（关注方向一致性）
    - Magnitude: Smooth L1 Loss
    """
    def __init__(self, src_model_path, tgt_model_path, common_vocab=None, converter_type="bp", 
                 lr=1e-2, num_iterations=2000, lambd=0.99):
        super().__init__(src_model_path, tgt_model_path, common_vocab, converter_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.src_anchor_embeddings, self.tgt_anchor_embeddings = self._get_common_embeddings(common_vocab)
        # 保持 embeddings 在 CPU 上，按需移到 GPU 以节省内存
        self.src_embeddings = self._get_embeddings(src_model_path).cpu()
        self.tgt_embeddings = self._get_embeddings(tgt_model_path).cpu()
        self.lr = lr
        self.num_iterations = num_iterations
        self.lambd = lambd

    def _embedding2relative(self, embedding, anchor_embeddings):
        '''
        Convert an embedding to a relative embedding.
        '''
        A = embedding.to(torch.float32).to(self.device)
        B = anchor_embeddings.to(torch.float32).to(self.device)

        # Ensure A is 2D: [batch_size, embedding_dim]
        if A.dim() == 1:
            A = A.unsqueeze(0)

        # Cosine similarity: [batch_size, n_anchor]
        direction_similarity = F.normalize(A, dim=-1) @ F.normalize(B, dim=-1).t()
        
        A_norm = torch.norm(A, dim=-1, keepdim=True)  # [batch_size, 1]
        B_norm = torch.norm(B, dim=-1, keepdim=True).t()  # [1, n_anchors]

        # Broadcast: [batch_size, 1] / [1, n_anchors] -> [batch_size, n_anchors]
        magnitude_encoding = torch.log(A_norm + 1e-12) - torch.log(B_norm + 1e-12)

        return {
            "direction_similarity": direction_similarity,
            "magnitude_encoding": magnitude_encoding
        }
        
    def _initialization(self, shape, tgt_anchor_embeddings):
        '''
        随机初始化：使用目标 anchor 的均值和方差
        '''
        candidate_embeddings = torch.empty(shape, dtype=torch.float32).to(self.device)
        tgt_mean = torch.mean(tgt_anchor_embeddings, dim=0)
        tgt_std = torch.std(tgt_anchor_embeddings, dim=0)
        mean_val = float(torch.mean(tgt_mean))
        std_val = float(torch.mean(tgt_std))
        with torch.no_grad():
            candidate_embeddings.normal_(mean=mean_val, std=std_val)
        return candidate_embeddings

    def _search(self, embedding):
        embedding = embedding.to(torch.float32).to(self.device)

        with torch.enable_grad():
            src_rels = self._embedding2relative(embedding, self.src_anchor_embeddings)

            # 初始化候选 embedding
            candidate_embeddings = self._initialization((embedding.shape[0], self.tgt_anchor_embeddings.shape[1]), self.tgt_anchor_embeddings)

            candidate_embeddings = candidate_embeddings.clone().detach().requires_grad_(True)

            # 优化搜索
            optimizer = torch.optim.AdamW([candidate_embeddings], lr=self.lr)
            best_loss = 1e9
            best_candidate_embeddings = None
            best_index = None
            
            for i in range(self.num_iterations):
                x_rel = self._embedding2relative(candidate_embeddings, self.tgt_anchor_embeddings)

                # Cosine loss for direction
                direction_loss = 1 - F.cosine_similarity(
                    x_rel['direction_similarity'], 
                    src_rels['direction_similarity'], 
                    dim=-1
                ).mean()
                
                # Smooth L1 loss for magnitude
                magnitude_loss = F.smooth_l1_loss(
                    x_rel['magnitude_encoding'], 
                    src_rels['magnitude_encoding']
                )

                loss = self.lambd * direction_loss + (1 - self.lambd) * magnitude_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if i % 50 == 0:
                    print(f"loss: {loss.item()}, direction loss: {direction_loss.item()}, magnitude loss: {magnitude_loss.item()}, learning rate: {optimizer.param_groups[0]['lr']}")
                    if loss < best_loss:
                        best_loss = loss
                        best_candidate_embeddings = candidate_embeddings.clone()
                        best_index = i
                        
            if DEBUG:
                print(f"  Best loss: {best_loss.item():.6f}, Best index: {best_index}")

            return best_candidate_embeddings

    def convert(self, embedding):
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        
        converted_embedding = self._search(embedding)
        return converted_embedding


if __name__ == "__main__":
    converter = BPConverter(
        src_model_path="/data/Llama-3.2-1B-Instruct",
        tgt_model_path="/data/Llama-3.2-3B-Instruct"
    )
    
    print("=" * 80)
    print("Anchor Embeddings Info:")
    print(f"  Source anchor embeddings shape: {converter.src_anchor_embeddings.shape}")
    print(f"  Target anchor embeddings shape: {converter.tgt_anchor_embeddings.shape}")
    print(f"  Number of anchors: {converter.src_anchor_embeddings.shape[0]}")
    print("=" * 80)
    
    test_tokens = ["hello", "world", "AI"]
    src_tokenizer = AutoTokenizer.from_pretrained(converter.src_model_path)
    tgt_tokenizer = AutoTokenizer.from_pretrained(converter.tgt_model_path)

    for token in test_tokens:
        print(f"\n{'='*80}")
        print(f"Testing token: '{token}'")
        print("=" * 80)
        
        src_token_id = src_tokenizer.convert_tokens_to_ids(token)
        tgt_token_id = tgt_tokenizer.convert_tokens_to_ids(token)
        print(f"Source Token ID: {src_token_id}, Target Token ID: {tgt_token_id}")
        
        # 计算源 token 相对于 src_anchors 的相对表示
        src_embedding = converter.src_embeddings[src_token_id]
        src_rel = converter._embedding2relative(src_embedding, converter.src_anchor_embeddings)
        
        # 计算目标 token 相对于 tgt_anchors 的相对表示
        tgt_embedding = converter.tgt_embeddings[tgt_token_id]
        tgt_rel = converter._embedding2relative(tgt_embedding, converter.tgt_anchor_embeddings)
        
        # 计算相对表示向量的余弦相似度
        direction_cosine = F.cosine_similarity(
            src_rel['direction_similarity'], 
            tgt_rel['direction_similarity'], 
            dim=1
        ).item()
        magnitude_l1 = F.l1_loss(
            src_rel['magnitude_encoding'], 
            tgt_rel['magnitude_encoding']
        ).item()
        print(f"[Source vs Target] Relative Representation:")
        print(f"  Direction cosine similarity: {direction_cosine:.6f}")
        print(f"  Magnitude L1 loss: {magnitude_l1:.6f}")
        
        # 执行 BPSearch 并检查结果
        print(f"\nRunning BPSearch...")
        converted_embedding = converter.convert(src_embedding)
        
        # 检查转换后的 embedding 与目标 embedding 的相似度
        converted_cosine = F.cosine_similarity(
            converted_embedding,
            tgt_embedding.unsqueeze(0).to(converter.device),
            dim=1
        ).item()
        print(f"Converted embedding cosine similarity with target: {converted_cosine:.6f}")
    
    print("\n" + "=" * 80)
    print("Finding similar tokens in target vocabulary...")
    print("=" * 80)
    find_similar_tokens(converter, test_tokens)
