"""
MLPSupervisedConverter: 监督 MLP 转换器

使用配对的源-目标 embeddings 进行监督训练：
- 使用 common vocab 中的 token 对作为训练数据
- 支持加载配对的压缩向量（源和目标模型）
- 直接最小化转换后的 embedding 与目标 embedding 之间的距离
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from base_converter import Converter, find_similar_tokens
from transformers import AutoTokenizer

DEBUG = True


class MLP(nn.Module):
    """可配置隐藏层的 MLP，支持Dropout正则化"""
    def __init__(self, input_dim, output_dim, hidden_layers, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=True))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim, bias=True))
        layers.append(nn.LayerNorm(output_dim, elementwise_affine=True))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class MLPSupervisedConverter(Converter):
    """
    监督 MLP 转换器
    
    使用配对的源-目标 embeddings 进行监督训练。
    训练目标：学习一个 MLP，使转换后的 embedding 直接逼近目标 embedding。
    
    训练数据来源：
    - common vocab 的 token embeddings（源-目标配对）
    - 压缩向量（源-目标配对，可选）
    """
    def __init__(self, src_model_path, tgt_model_path, common_vocab=None, converter_type="mlp_st",
                 hidden_layers=None, lr=1e-3, epochs=100, batch_size=256, lambd=0.5,
                 src_compression_file=None, tgt_compression_file=None,
                 eval_ratio=0.1, eval_interval=10, save_dir=None, load_path=None,
                 dropout=0.1, weight_decay=0.01, max_grad_norm=1.0, 
                 early_stopping_patience=50, early_stopping_min_delta=1e-6):
        super().__init__(src_model_path, tgt_model_path, common_vocab, converter_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 超参数
        self.hidden_layers = hidden_layers if hidden_layers is not None else [2048]
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambd = lambd  # cosine loss 权重 (1 - lambd 为 MSE loss 权重)
        self.eval_ratio = eval_ratio
        self.eval_interval = eval_interval
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        
        # 保存路径和日志设置
        self.save_dir = Path(save_dir) if save_dir else Path("./outputs/mlp_st")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logger()
        
        # 获取配对的 embeddings（监督学习可以使用目标 embeddings）
        self.src_anchor_embeddings, self.tgt_anchor_embeddings = self._get_common_embeddings(common_vocab)
        self.src_anchor_embeddings = self.src_anchor_embeddings.to(self.device)
        self.tgt_anchor_embeddings = self.tgt_anchor_embeddings.to(self.device)
        
        # 获取维度信息
        self.src_dim = self.src_anchor_embeddings.shape[1]
        self.tgt_dim = self.tgt_anchor_embeddings.shape[1]
        
        # 加载压缩向量（可选，需要配对）
        self.src_compression_vectors = None
        self.tgt_compression_vectors = None
        if src_compression_file and tgt_compression_file:
            self.src_compression_vectors, self.tgt_compression_vectors = \
                self._load_paired_compression_vectors(src_compression_file, tgt_compression_file)
        
        # 如果提供了 load_path，则加载已保存的模型
        if load_path:
            self.logger.info(f"Loading converter from: {load_path}")
            checkpoint = torch.load(load_path, map_location=self.device)
            # 从 checkpoint 恢复配置
            self.hidden_layers = checkpoint.get('hidden_layers', self.hidden_layers)
            self.src_dim = checkpoint.get('src_dim', self.src_dim)
            self.tgt_dim = checkpoint.get('tgt_dim', self.tgt_dim)
            self.dropout = checkpoint.get('dropout', self.dropout)
            # 初始化 MLP
            self.mlp = MLP(self.src_dim, self.tgt_dim, self.hidden_layers, dropout=self.dropout).to(self.device)
            # 加载模型权重
            self.mlp.load_state_dict(checkpoint['mlp_state_dict'])
            self.mlp.eval()
            self.logger.info(f"✓ Converter loaded successfully from {load_path}")
        else:
            # 初始化 MLP
            self.mlp = MLP(self.src_dim, self.tgt_dim, self.hidden_layers, dropout=self.dropout).to(self.device)
            
            # 准备训练数据
            self.train_loader, self.eval_loader = self._prepare_data()

            print(len(self.train_loader.dataset))
            
            # 训练模型
            self._train()
    
    def _setup_logger(self):
        """设置日志记录器"""
        log_file = self.save_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        self.logger = logging.getLogger(f"MLPSupervisedConverter_{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # 清除已有 handlers
        
        # 文件 handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # 控制台 handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def _load_paired_compression_vectors(self, src_file, tgt_file):
        """
        加载配对的压缩向量（源和目标模型）
        
        要求源和目标的压缩结果是对同一批文本进行的，以 text_id 进行匹配
        """
        with open(src_file, 'r', encoding='utf-8') as f:
            src_data = json.load(f)
        with open(tgt_file, 'r', encoding='utf-8') as f:
            tgt_data = json.load(f)
        
        # 按 text_id 构建索引
        src_dict = {r['text_id']: r['memory'] for r in src_data.get('results', [])}
        tgt_dict = {r['text_id']: r['memory'] for r in tgt_data.get('results', [])}
        
        # 找到共同的 text_ids
        common_ids = set(src_dict.keys()) & set(tgt_dict.keys())
        self.logger.info(f"Found {len(common_ids)} paired compression vectors")
        
        src_vectors = []
        tgt_vectors = []
        
        for text_id in common_ids:
            src_memory = torch.tensor(src_dict[text_id], dtype=torch.float32)
            tgt_memory = torch.tensor(tgt_dict[text_id], dtype=torch.float32)
            
            # 展平 memory tokens: [n_mem_tokens, dim] -> n_mem_tokens 个 [dim] 向量
            if src_memory.dim() == 2 and tgt_memory.dim() == 2:
                # 确保 memory token 数量一致
                n_tokens = min(src_memory.shape[0], tgt_memory.shape[0])
                for i in range(n_tokens):
                    src_vectors.append(src_memory[i])
                    tgt_vectors.append(tgt_memory[i])
            elif src_memory.dim() == 1 and tgt_memory.dim() == 1:
                src_vectors.append(src_memory)
                tgt_vectors.append(tgt_memory)
        
        if src_vectors:
            return torch.stack(src_vectors), torch.stack(tgt_vectors)
        return None, None
    
    def _prepare_data(self):
        """准备训练和验证数据（监督，需要源-目标配对）"""
        # 获取token embeddings（限制数量以避免过拟合）
        src_emb = self.src_anchor_embeddings.cpu().to(torch.float32)
        tgt_emb = self.tgt_anchor_embeddings.cpu().to(torch.float32)
        
        # 如果数据量太大，可以限制使用前N个（可选）
        # 注意：这里应该限制tensor的行数，而不是列表
        # max_token_pairs = 100  # 设置为None使用全部数据，或设置具体数值如5000
        # if max_token_pairs is not None and src_emb.shape[0] > max_token_pairs:
        #     src_emb = src_emb[:max_token_pairs]
        #     tgt_emb = tgt_emb[:max_token_pairs]
        #     self.logger.info(f"Limited token embeddings to {max_token_pairs} pairs")
        
        # 初始化数据列表，首先添加token embeddings
        all_src_data = [src_emb[:1]]
        all_tgt_data = [tgt_emb[:1]]

        self.logger.info(f"Token embedding pairs: {src_emb.shape[0]}")
        
        # 添加压缩向量（如果存在）
        if self.src_compression_vectors is not None and self.tgt_compression_vectors is not None:
            all_src_data.append(self.src_compression_vectors.cpu().to(torch.float32))
            all_tgt_data.append(self.tgt_compression_vectors.cpu().to(torch.float32))
            self.logger.info(f"Compression vector pairs: {self.src_compression_vectors.shape[0]}")
        
        # 合并数据
        if len(all_src_data) == 0:
            raise ValueError("No training data available. Need at least token embeddings or compression vectors.")
        
        src_data = torch.cat(all_src_data, dim=0)
        tgt_data = torch.cat(all_tgt_data, dim=0)
        self.logger.info(f"Total training pairs: {src_data.shape[0]}")
        
        # 划分训练集和验证集
        dataset = TensorDataset(src_data, tgt_data)
        n_total = len(dataset)
        n_eval = int(n_total * self.eval_ratio)
        n_train = n_total - n_eval
        
        train_dataset, eval_dataset = random_split(dataset, [n_train, n_eval])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.logger.info(f"Train pairs: {n_train}, Eval pairs: {n_eval}")
        
        return train_loader, eval_loader
    
    def _compute_loss(self, converted_embeddings, tgt_embeddings):
        """
        计算监督损失：转换后的 embedding 应接近目标 embedding
        """
        # Cosine loss: 1 - cosine_similarity（方向一致性）
        cosine_loss = 1 - F.cosine_similarity(converted_embeddings, tgt_embeddings, dim=-1).mean()
        
        # MSE loss（幅度和方向）
        mse_loss = F.mse_loss(converted_embeddings, tgt_embeddings)
        
        loss = self.lambd * cosine_loss + (1 - self.lambd) * mse_loss
        return loss, cosine_loss, mse_loss
    
    def _evaluate(self):
        """在验证集上评估"""
        self.mlp.eval()
        total_loss = 0
        total_cos_loss = 0
        total_mse_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(self.eval_loader, desc="Evaluating", leave=False, ncols=100)
            for src_batch, tgt_batch in pbar:
                src_batch = src_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)
                converted = self.mlp(src_batch)
                loss, cos_loss, mse_loss = self._compute_loss(converted, tgt_batch)
                
                total_loss += loss.item()
                total_cos_loss += cos_loss.item()
                total_mse_loss += mse_loss.item()
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'cos': f'{cos_loss.item():.4f}',
                    'mse': f'{mse_loss.item():.4f}'
                })
        
        n_batches = len(self.eval_loader)
        return total_loss / n_batches, total_cos_loss / n_batches, total_mse_loss / n_batches
    
    def _train(self):
        """训练 MLP"""
        self.logger.info("=" * 80)
        self.logger.info("Training MLPSupervisedConverter...")
        self.logger.info(f"  Hidden layers: {self.hidden_layers}")
        self.logger.info(f"  Learning rate: {self.lr}")
        self.logger.info(f"  Epochs: {self.epochs}")
        self.logger.info(f"  Batch size: {self.batch_size}")
        self.logger.info(f"  Lambda (cosine weight): {self.lambd}")
        self.logger.info(f"  Eval interval: {self.eval_interval}")
        self.logger.info(f"  Dropout: {self.dropout}")
        self.logger.info(f"  Weight decay: {self.weight_decay}")
        self.logger.info(f"  Max grad norm: {self.max_grad_norm}")
        self.logger.info(f"  Early stopping patience: {self.early_stopping_patience}")
        self.logger.info("=" * 80)
        
        optimizer = torch.optim.AdamW(self.mlp.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        best_eval_loss = float('inf')
        best_state = None
        training_log = []
        patience_counter = 0  # Early stopping计数器
        
        # 外层 epoch 进度条
        epoch_pbar = tqdm(range(self.epochs), desc="Training", ncols=120)
        
        for epoch in epoch_pbar:
            self.mlp.train()
            total_loss = 0
            total_cos_loss = 0
            total_mse_loss = 0
            
            # 内层 batch 进度条
            batch_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", 
                             leave=False, ncols=100)
            
            for src_batch, tgt_batch in batch_pbar:
                src_batch = src_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)
                
                # 前向传播
                converted = self.mlp(src_batch)
                
                # 计算损失（监督：直接与目标 embedding 比较）
                loss, cos_loss, mse_loss = self._compute_loss(converted, tgt_batch)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.mlp.parameters(), self.max_grad_norm)
                optimizer.step()
                
                total_loss += loss.item()
                total_cos_loss += cos_loss.item()
                total_mse_loss += mse_loss.item()
                
                # 更新 batch 进度条
                batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'cos': f'{cos_loss.item():.4f}',
                    'mse': f'{mse_loss.item():.4f}'
                })
            
            scheduler.step()
            
            n_batches = len(self.train_loader)
            avg_loss = total_loss / n_batches
            avg_cos_loss = total_cos_loss / n_batches
            avg_mse_loss = total_mse_loss / n_batches
            
            # 定期评估
            if (epoch + 1) % self.eval_interval == 0:
                eval_loss, eval_cos_loss, eval_mse_loss = self._evaluate()
                
                log_entry = {
                    'epoch': epoch + 1,
                    'train_loss': avg_loss,
                    'train_cos_loss': avg_cos_loss,
                    'train_mse_loss': avg_mse_loss,
                    'eval_loss': eval_loss,
                    'eval_cos_loss': eval_cos_loss,
                    'eval_mse_loss': eval_mse_loss,
                    'lr': scheduler.get_last_lr()[0]
                }
                training_log.append(log_entry)
                
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.epochs}: "
                    f"train_loss={avg_loss:.6f}, eval_loss={eval_loss:.6f}, "
                    f"train_dir_loss={avg_cos_loss:.6f}, train_mag_loss={avg_mse_loss:.6f}, "
                    f"eval_dir_loss={eval_cos_loss:.6f}, eval_mag_loss={eval_mse_loss:.6f}, "
                    f"lr={scheduler.get_last_lr()[0]:.6f}"
                )
                
                # Early stopping检查
                if eval_loss < best_eval_loss - self.early_stopping_min_delta:
                    best_eval_loss = eval_loss
                    best_state = self.mlp.state_dict().copy()
                    patience_counter = 0  # 重置计数器
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        self.logger.info(
                            f"Early stopping triggered at epoch {epoch + 1}. "
                            f"No improvement for {self.early_stopping_patience} evaluations. "
                            f"Best eval loss: {best_eval_loss:.6f}"
                        )
                        break
                
                # 更新 epoch 进度条
                epoch_pbar.set_postfix({
                    'train_loss': f'{avg_loss:.4f}',
                    'eval_loss': f'{eval_loss:.4f}',
                    'best_eval': f'{best_eval_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })
            else:
                # 更新 epoch 进度条（无评估时）
                epoch_pbar.set_postfix({
                    'train_loss': f'{avg_loss:.4f}',
                    'best_eval': f'{best_eval_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })
        
        # 恢复最佳模型
        if best_state is not None:
            self.mlp.load_state_dict(best_state)
        
        self.logger.info(f"Training completed. Best eval loss: {best_eval_loss:.6f}")
        
        # 保存训练日志
        log_file = self.save_dir / "training_log.json"
        with open(log_file, 'w') as f:
            json.dump(training_log, f, indent=2)
        
        self.mlp.eval()
    
    def convert(self, embedding):
        """将源 embedding 转换为目标空间"""
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        
        embedding = embedding.to(torch.float32).to(self.device)
        
        with torch.no_grad():
            converted = self.mlp(embedding)
        
        return converted
    
    def save(self, path=None):
        """保存模型"""
        if path is None:
            path = self.save_dir / "model.pt"
        torch.save({
            'mlp_state_dict': self.mlp.state_dict(),
            'hidden_layers': self.hidden_layers,
            'src_dim': self.src_dim,
            'tgt_dim': self.tgt_dim,
            'dropout': self.dropout,
        }, path)
        self.logger.info(f"Model saved to {path}")
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.mlp.load_state_dict(checkpoint['mlp_state_dict'])
        self.mlp.eval()


if __name__ == "__main__":
    converter = MLPSupervisedConverter(
        src_model_path="/data/Llama-3.2-1B-Instruct",
        tgt_model_path="/data/Llama-3.2-3B-Instruct",
        hidden_layers=[4096],
        lr=5e-3,
        epochs=500,
        batch_size=16,
        eval_ratio=0.1,
        eval_interval=5,
        save_dir="./outputs/mlp_st_test_dropout",
        # 可选：加载压缩向量
        src_compression_file="./outputs/mem_compress_from_token/Llama-3.2-1B-Instruct_mem4_len256_64.0x/compression_results.json",
        tgt_compression_file="./outputs/mem_compress_from_token/Llama-3.2-3B-Instruct_mem4_len256_64.0x/compression_results.json",
    )
    
    converter.save()
    
    print("\n" + "=" * 80)
    print("Testing conversion...")
    print("=" * 80)
    
    test_tokens = ["hello", "world", "AI"]
    find_similar_tokens(converter, test_tokens)
