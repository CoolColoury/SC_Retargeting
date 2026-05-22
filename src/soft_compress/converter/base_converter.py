'''
This module contains the base Converter class for all converters.
'''

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from tqdm import tqdm

def find_similar_tokens(converter, src_tokens, top_k=5):
    # 获取device，保证所有Tensor在相同device
    device = converter.device if hasattr(converter, 'device') else torch.device("cpu")

    # 加载tokenizer
    src_tokenizer = AutoTokenizer.from_pretrained(converter.src_model_path)
    tgt_tokenizer = AutoTokenizer.from_pretrained(converter.tgt_model_path)

    # 获取源token的embedding
    src_model = AutoModelForCausalLM.from_pretrained(converter.src_model_path)
    src_embeddings = src_model.get_input_embeddings().weight.detach().to(device)

    # 获取目标模型的完整词表embedding
    tgt_model = AutoModelForCausalLM.from_pretrained(converter.tgt_model_path)
    tgt_embeddings = tgt_model.get_input_embeddings().weight.detach().to(device)

    # 对每个源token进行处理
    for token in src_tokens:
        # 获取token的id和embedding
        token_id = src_tokenizer.convert_tokens_to_ids(token)
        if token_id == src_tokenizer.unk_token_id:
            print(f"警告: token '{token}' 在源模型词表中未找到")
            continue

        token_embedding = src_embeddings[token_id]

        # 转换embedding并确保device一致
        converted_embedding = converter.convert(token_embedding)
        converted_embedding = converted_embedding.to(device)

        # 计算与目标词表中所有token的余弦相似度
        similarity = torch.nn.functional.cosine_similarity(
            converted_embedding,  # [tgt_dim]
            tgt_embeddings,
            dim=1
        )

        # 获取top_k个最相似的token
        top_values, top_indices = torch.topk(similarity, k=top_k)

        print(f"\n源token: {token}")
        for i in range(top_k):
            similar_token = tgt_tokenizer.convert_ids_to_tokens(int(top_indices[i]))
            print(f"相似token: {similar_token}, 相似度: {float(top_values[i]):.4f}")

    # 清理内存
    del src_model
    del tgt_model
    torch.cuda.empty_cache()

class Converter:
    '''
    This class is the base class for all converters.
    '''
    def __init__(self, src_model_path, tgt_model_path, common_vocab, converter_type, **kwargs):
        '''
        Initialize the converter class.
        '''
        self.src_model_path = src_model_path
        self.tgt_model_path = tgt_model_path
        self.common_vocab = common_vocab
        self.converter_type = converter_type
        self.kwargs = kwargs

    def convert(self, data):
        '''
        Convert the data to the desired format.
        '''
        raise NotImplementedError("Subclasses must implement this method")
    
    def _build_vocab(self, tokenizer):
        vocab = list(tokenizer.get_vocab())
        inverse_vocab = {w: i for i, w in enumerate(vocab)}
        return vocab, inverse_vocab

    def _vocab_cleaning(self, vocab, enhanced: bool = False):
        """
        Basic vocabulary cleaning.
        
        Args:
            vocab: List of tokens
            enhanced: If True, use enhanced filtering (requires vocab_filter module)
        """
        special_tokens = ['<bos>', '<eos>', '<pad>', '<unk>', '<s>', '</s>',
        '<sep>', '<cls>', '<mask>', '<eot>', '<|endoftext|>', 
        '<|startoftext|>', '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[EOT]']
        cleaned = []
        for token in vocab:
            if token in special_tokens:
                continue
            if token.startswith('<') and token.endswith('>'):
                continue
            if token.startswith('[') and token.endswith(']'):
                continue
            if len(token.strip()) == 0:
                continue
            cleaned.append(token)
        return cleaned
    
    def _get_vocab(self, tokenizer):
        vocab, _ = self._build_vocab(tokenizer)
        vocab = self._vocab_cleaning(vocab)
        return vocab

    def _get_embeddings(self, model_name_or_path):
        model_name_lower = model_name_or_path.lower()
        if ('qwen' in model_name_lower or 
            'mistral' in model_name_lower or 
            'llama' in model_name_lower):
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            embeddings = model.get_input_embeddings().weight.detach().cpu()
            del model
            return embeddings
        raise NotImplementedError(f'Model type not supported yet! Model path: {model_name_or_path}')
    
    def _vocab_filter(self, common_vocab):
        src_tokenizer = AutoTokenizer.from_pretrained(self.src_model_path)
        tgt_tokenizer = AutoTokenizer.from_pretrained(self.tgt_model_path)

        if common_vocab is not None:
            with open(common_vocab, "r", encoding="utf-8") as f:
                filtered_vocab = [x.strip() for x in f.readlines()]
        else:
            src_vocab = self._get_vocab(src_tokenizer)
            tgt_vocab = self._get_vocab(tgt_tokenizer)
            filtered_vocab = list(set(src_vocab) & set(tgt_vocab))
            print(f"Number of common tokens: {len(filtered_vocab)}")

        src_indices = []
        tgt_indices = []

        for word in filtered_vocab:
            src_token = src_tokenizer.tokenize(word)
            tgt_token = tgt_tokenizer.tokenize(word)
            if len(src_token) == len(tgt_token) == 1:
                src_idx = src_tokenizer.convert_tokens_to_ids(src_token)[0]
                tgt_idx = tgt_tokenizer.convert_tokens_to_ids(tgt_token)[0]

                if src_idx != src_tokenizer.unk_token_id and tgt_idx != tgt_tokenizer.unk_token_id:
                    src_indices.append(src_idx)
                    tgt_indices.append(tgt_idx)

        assert len(src_indices) == len(tgt_indices)
        print(f"Number of common tokens: {len(src_indices)}")
        return src_indices, tgt_indices
    
    def _vocab_filter_v2(self, common_vocab):
        SUBWORD_PREFIXES = ['Ġ', '▁', '##', '@@']
        
        SPECIAL_TOKENS = [
            '<bos>', '<eos>', '<pad>', '<unk>', '<s>', '</s>',
            '<sep>', '<cls>', '<mask>', '<eot>', '<|endoftext|>',
            '<|startoftext|>', '[PAD]', '[UNK]', '[CLS]', '[SEP]', 
            '[MASK]', '[EOT]', '<|im_start|>', '<|im_end|>'
        ]

        # 不使用 NLTK，改用简单的启发式方法过滤 token
        # 只保留看起来像英语单词的 token（只包含字母，长度合理）

        print("  Loading tokenizers...")
        src_tokenizer = AutoTokenizer.from_pretrained(self.src_model_path)
        tgt_tokenizer = AutoTokenizer.from_pretrained(self.tgt_model_path)
        print("  Tokenizers loaded")

        print("  Building source vocabulary...")
        src_vocab = self._get_vocab(src_tokenizer)
        print(f"  Source vocab size: {len(src_vocab)}")
        
        print("  Building target vocabulary...")
        tgt_vocab = self._get_vocab(tgt_tokenizer)
        print(f"  Target vocab size: {len(tgt_vocab)}")
        
        print("  Finding common tokens...")
        filtered_vocab = list(set(src_vocab) & set(tgt_vocab))
        print(f"  Number of raw common tokens: {len(filtered_vocab)}")

        def is_valid_token(token: str) -> bool:
            for prefix in SUBWORD_PREFIXES:
                if token.startswith(prefix):
                    return False
            if token in SPECIAL_TOKENS or token.lower() in [t.lower() for t in SPECIAL_TOKENS]:
                return False
            if token.startswith('<') and token.endswith('>'):
                return False
            if token.startswith('[') and token.endswith(']'):
                return False
            if len(token.strip()) == 0:
                return False
            if token.isdigit():
                return False
            # 使用简单的启发式方法：只保留看起来像英语单词的 token
            # 1. 只包含字母（允许大小写）
            # 2. 长度在合理范围内（2-20个字符）
            # 3. 至少包含一个元音字母（a, e, i, o, u）
            token_clean = token.strip()
            if len(token_clean) < 2 or len(token_clean) > 20:
                return False
            if not token_clean.isalpha():
                return False
            # 检查是否包含元音字母
            vowels = set('aeiouAEIOU')
            if not any(c in vowels for c in token_clean):
                return False
            return True
        
        print("  Filtering tokens (using heuristic filtering)...")
        filtered_vocab = [token for token in tqdm(filtered_vocab, desc="    Filtering", leave=False) if is_valid_token(token)]
        print(f"  Number of filtered common tokens: {len(filtered_vocab)}")

        print("  Mapping tokens to indices...")
        src_indices = []
        tgt_indices = []

        for word in tqdm(filtered_vocab, desc="    Mapping", leave=False):
            src_token = src_tokenizer.tokenize(word)
            tgt_token = tgt_tokenizer.tokenize(word)
            if len(src_token) == len(tgt_token) == 1:
                src_idx = src_tokenizer.convert_tokens_to_ids(src_token)[0]
                tgt_idx = tgt_tokenizer.convert_tokens_to_ids(tgt_token)[0]

                if src_idx != src_tokenizer.unk_token_id and tgt_idx != tgt_tokenizer.unk_token_id:
                    src_indices.append(src_idx)
                    tgt_indices.append(tgt_idx)

        assert len(src_indices) == len(tgt_indices)
        print(f"  Number of final anchor tokens: {len(src_indices)}")
        return src_indices, tgt_indices

    def _vocab_filter_v3(self, common_vocab, similarity_threshold=0.98):
        """
        在 _vocab_filter_v2 的基础上，基于相对表示的相似度筛选 anchors
        
        这种方法使用相对表示（relative representation）来计算相似度，
        即每个 token 相对于其他 anchors 的表示，而不是直接比较 embeddings。
        
        Args:
            common_vocab: Common vocabulary file path or None
            similarity_threshold: Minimum cosine similarity threshold (default: 0.98)
        
        Returns:
            src_indices: List[int] - 源模型anchor token IDs (filtered by similarity)
            tgt_indices: List[int] - 目标模型anchor token IDs (filtered by similarity)
        """
        # 先使用 _vocab_filter_v2 获取候选 anchors
        print("  Step 1: Getting candidate anchors using vocab filtering...")
        src_indices, tgt_indices = self._vocab_filter_v2(common_vocab)
        
        if len(src_indices) == 0:
            print("  Warning: No candidate anchors found!")
            return src_indices, tgt_indices
        
        # 加载 embeddings
        print("  Step 2: Loading model embeddings for similarity check...")
        src_embeddings = self._get_embeddings(self.src_model_path)
        tgt_embeddings = self._get_embeddings(self.tgt_model_path)
        
        # 获取候选 anchors 的 embeddings
        src_anchor_embeds = src_embeddings[src_indices]  # [num_candidates, src_embed_dim]
        tgt_anchor_embeds = tgt_embeddings[tgt_indices]  # [num_candidates, tgt_embed_dim]
        
        src_dim = src_anchor_embeds.shape[1]
        tgt_dim = tgt_anchor_embeds.shape[1]
        
        # 计算基于相对表示的相似度
        print(f"  Step 3: Computing relative representation similarity for {len(src_indices)} candidate anchors...")
        print(f"    Source embedding dim: {src_dim}, Target embedding dim: {tgt_dim}")
        print(f"    Using relative representation (direction cosine similarity)...")
        
        # 对每个候选 anchor，计算它相对于所有其他候选 anchors 的相对表示
        # 相对表示 = 归一化后与其他 anchors 的余弦相似度
        
        # 归一化 embeddings
        src_norm = F.normalize(src_anchor_embeds, dim=1)  # [num_candidates, src_dim]
        tgt_norm = F.normalize(tgt_anchor_embeds, dim=1)  # [num_candidates, tgt_dim]
        
        # 计算相对表示矩阵（每行是一个 token 相对于所有 anchors 的余弦相似度）
        src_relative = torch.matmul(src_norm, src_norm.T)  # [num_candidates, num_candidates]
        tgt_relative = torch.matmul(tgt_norm, tgt_norm.T)  # [num_candidates, num_candidates]
        
        # 计算每个 token 的相对表示向量之间的余弦相似度
        similarities = []
        for i in tqdm(range(len(src_indices)), desc="    Computing similarities", leave=False):
            # 获取第 i 个 token 的相对表示
            src_rel_i = src_relative[i]  # [num_candidates]
            tgt_rel_i = tgt_relative[i]  # [num_candidates]
            
            # 计算余弦相似度
            sim = F.cosine_similarity(src_rel_i.unsqueeze(0), tgt_rel_i.unsqueeze(0), dim=1).item()
            similarities.append(sim)
        
        similarities = torch.tensor(similarities)
        
        # 固定阈值：使用用户指定的阈值（默认 0.98）
        max_similarity = similarities.max().item()
        min_similarity = similarities.min().item()
        mean_similarity = similarities.mean().item()
        
        print(f"  Step 4: Similarity statistics - min: {min_similarity:.4f}, mean: {mean_similarity:.4f}, max: {max_similarity:.4f}")
        print(f"  Using fixed threshold: {similarity_threshold:.4f}")
        
        valid_mask = similarities >= similarity_threshold
        valid_indices = torch.where(valid_mask)[0]
        
        filtered_src_indices = [src_indices[i] for i in valid_indices.tolist()]
        filtered_tgt_indices = [tgt_indices[i] for i in valid_indices.tolist()]
        filtered_similarities = similarities[valid_indices].tolist()
        
        print(f"  Original anchors: {len(src_indices)}")
        print(f"  Filtered anchors (similarity >= {similarity_threshold:.4f}): {len(filtered_src_indices)}")
        if len(filtered_src_indices) > 0:
            print(f"  Similarity range: [{min(filtered_similarities):.4f}, {max(filtered_similarities):.4f}]")
            print(f"  Mean similarity: {sum(filtered_similarities) / len(filtered_similarities):.4f}")
        else:
            print(f"  Warning: No anchors passed the similarity threshold!")
            print(f"  Original similarity range: [{similarities.min().item():.4f}, {similarities.max().item():.4f}]")
            print(f"  Consider lowering the threshold or checking model compatibility.")
        
        assert len(filtered_src_indices) == len(filtered_tgt_indices)
        return filtered_src_indices, filtered_tgt_indices

    def _get_common_embeddings(self, common_vocab, use_similarity_filter=True, similarity_threshold=0.98):
        """
        获取 common vocab 的 embeddings
        
        Args:
            common_vocab: Common vocabulary file path or None
            use_similarity_filter: If True, use _vocab_filter_v3 to filter by similarity (default: True)
            similarity_threshold: Minimum cosine similarity threshold when use_similarity_filter=True (default: 0.98)
        
        Returns:
            src_common: Tensor - 源模型common embeddings
            tgt_common: Tensor - 目标模型common embeddings
        """
        # if use_similarity_filter:
        #     src_indices, tgt_indices = self._vocab_filter_v3(common_vocab, similarity_threshold=similarity_threshold)
        # else:
        #     src_indices, tgt_indices = self._vocab_filter_v2(common_vocab)
        
        src_indices, tgt_indices = self._vocab_filter(common_vocab)
        
        src_embeddings = self._get_embeddings(self.src_model_path)
        tgt_embeddings = self._get_embeddings(self.tgt_model_path)

        src_common = src_embeddings[src_indices]
        tgt_common = tgt_embeddings[tgt_indices]

        return src_common, tgt_common

