"""
MLPUnsupervisedConverter: 无监督 MLP 转换器

使用相对表示空间进行训练，不直接使用目标 embeddings：
- 训练 MLP 将源 embedding 的相对表示映射到目标空间的相对表示
- 使用 anchor embeddings 计算相对表示
- 训练数据包括：原始 embeddings、合成/插值 embeddings、压缩向量
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
    """可配置隐藏层的 MLP"""
    def __init__(self, input_dim, output_dim, hidden_layers):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class MLPUnsupervisedConverter(Converter):
    """
    无监督 MLP 转换器
    
    通过相对表示空间进行训练，不直接使用目标 embeddings。
    训练目标：学习一个 MLP，使转换后的 embedding 在目标 anchor 空间中的
              相对表示与源 embedding 在源 anchor 空间中的相对表示一致。
    
    训练数据来源：
    - 原始 token embeddings
    - 合成数据：e1+e2+...+en, (e1+e2+...+en)/n 等
    - 压缩向量（可选）
    """
    def __init__(self, src_model_path, tgt_model_path, common_vocab=None, converter_type="mlp_ust",
                 hidden_layers=None, lr=1e-3, epochs=100, batch_size=256, lambd=0.9,
                 src_compression_file=None, n_synthetic=10000, eval_ratio=0.1, 
                 eval_interval=10, save_dir=None, load_path=None):
        super().__init__(src_model_path, tgt_model_path, common_vocab, converter_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 超参数
        self.hidden_layers = hidden_layers if hidden_layers is not None else [2048]
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambd = lambd  # direction loss 权重
        self.eval_ratio = eval_ratio
        self.eval_interval = eval_interval
        self.n_synthetic = n_synthetic
        
        # 保存路径和日志设置
        self.save_dir = Path(save_dir) if save_dir else Path("./outputs/mlp_ust")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logger()
        
        # 获取 anchor embeddings
        self.src_anchor_embeddings, self.tgt_anchor_embeddings = self._get_common_embeddings(common_vocab)
        self.src_anchor_embeddings = self.src_anchor_embeddings.to(self.device)
        self.tgt_anchor_embeddings = self.tgt_anchor_embeddings.to(self.device)
        
        # 获取完整 embeddings 用于训练
        self.src_embeddings = self._get_embeddings(src_model_path).cpu()
        
        # 获取维度信息
        self.src_dim = self.src_embeddings.shape[1]
        self.tgt_dim = self.tgt_anchor_embeddings.shape[1]
        
        # 加载压缩向量（可选）
        self.src_compression_vectors = None
        if src_compression_file:
            self.src_compression_vectors = self._load_compression_vectors(src_compression_file)
        
        # 如果提供了 load_path，则加载已保存的模型
        if load_path:
            self.logger.info(f"Loading converter from: {load_path}")
            checkpoint = torch.load(load_path, map_location=self.device)
            # 从 checkpoint 恢复配置
            self.hidden_layers = checkpoint.get('hidden_layers', self.hidden_layers)
            self.src_dim = checkpoint.get('src_dim', self.src_dim)
            self.tgt_dim = checkpoint.get('tgt_dim', self.tgt_dim)
            # 初始化 MLP
            self.mlp = MLP(self.src_dim, self.tgt_dim, self.hidden_layers).to(self.device)
            # 加载模型权重
            self.mlp.load_state_dict(checkpoint['mlp_state_dict'])
            self.mlp.eval()
            self.logger.info(f"✓ Converter loaded successfully from {load_path}")
        else:
            # 初始化 MLP
            self.mlp = MLP(self.src_dim, self.tgt_dim, self.hidden_layers).to(self.device)
            
            # 准备训练数据
            self.train_loader, self.eval_loader = self._prepare_data()
            
            # 训练模型
            self._train()
    
    def _setup_logger(self):
        """设置日志记录器"""
        log_file = self.save_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        self.logger = logging.getLogger(f"MLPUnsupervisedConverter_{id(self)}")
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
    
    def _load_compression_vectors(self, compression_file):
        """加载压缩向量（仅源模型）"""
        with open(compression_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vectors = []
        for result in data.get('results', []):
            memory = result.get('memory', [])
            if memory:
                vectors.append(torch.tensor(memory, dtype=torch.float32))
        
        if vectors:
            # 每个 memory 可能是 [n_mem_tokens, dim]，展平为多个向量
            all_vectors = []
            for v in vectors:
                if v.dim() == 2:
                    all_vectors.extend([v[i] for i in range(v.shape[0])])
                else:
                    all_vectors.append(v)
            return torch.stack(all_vectors)
        return None
    
    def _generate_synthetic_data(self, embeddings, n_samples):
        """
        生成合成数据（无监督，只生成源空间的数据）
        
        合成方式：
        - sum: e1 + e2 + ... + en
        - mean: (e1 + e2 + ... + en) / n
        - weighted: w1*e1 + w2*e2 + ... + wn*en
        """
        n_embeddings = embeddings.shape[0]
        synthetic_src = []
        
        samples_per_type = n_samples // 3
        
        for _ in range(samples_per_type):
            # 随机选择 2-5 个 embeddings
            n = torch.randint(2, min(6, n_embeddings + 1), (1,)).item()
            indices = torch.randperm(n_embeddings)[:n]
            selected = embeddings[indices]
            
            # Sum: e1 + e2 + ... + en
            synthetic_src.append(selected.sum(dim=0))
        
        for _ in range(samples_per_type):
            n = torch.randint(2, min(6, n_embeddings + 1), (1,)).item()
            indices = torch.randperm(n_embeddings)[:n]
            selected = embeddings[indices]
            
            # Mean: (e1 + e2 + ... + en) / n
            synthetic_src.append(selected.mean(dim=0))
        
        for _ in range(n_samples - 2 * samples_per_type):
            n = torch.randint(2, min(6, n_embeddings + 1), (1,)).item()
            indices = torch.randperm(n_embeddings)[:n]
            selected = embeddings[indices]
            
            # Weighted: w1*e1 + w2*e2 + ... + wn*en
            weights = torch.rand(n)
            weights = weights / weights.sum()
            synthetic_src.append((selected * weights.unsqueeze(1)).sum(dim=0))
        
        return torch.stack(synthetic_src)
    
    def _prepare_data(self):
        """准备训练和验证数据（无监督，只有源数据）"""
        all_src_data = [self.src_embeddings.to(torch.float32)]
        
        # 添加合成数据
        if self.n_synthetic > 0:
            synthetic_src = self._generate_synthetic_data(self.src_embeddings, self.n_synthetic)
            all_src_data.append(synthetic_src)
            self.logger.info(f"Generated {self.n_synthetic} synthetic samples")
        
        # 添加压缩向量
        if self.src_compression_vectors is not None:
            all_src_data.append(self.src_compression_vectors.to(torch.float32))
            self.logger.info(f"Loaded {self.src_compression_vectors.shape[0]} compression vectors")
        
        # 合并数据
        src_data = torch.cat(all_src_data, dim=0)
        self.logger.info(f"Total training samples: {src_data.shape[0]}")
        
        # 划分训练集和验证集
        dataset = TensorDataset(src_data)
        n_total = len(dataset)
        n_eval = int(n_total * self.eval_ratio)
        n_train = n_total - n_eval
        
        train_dataset, eval_dataset = random_split(dataset, [n_train, n_eval])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.logger.info(f"Train samples: {n_train}, Eval samples: {n_eval}")
        
        return train_loader, eval_loader
    
    def _embedding2relative(self, embedding, anchor_embeddings):
        """将 embedding 转换为相对于 anchor 的表示"""
        A = embedding.to(torch.float32)
        B = anchor_embeddings.to(torch.float32)
        
        if A.dim() == 1:
            A = A.unsqueeze(0)
        
        # 方向相似度: [batch_size, n_anchor]
        direction_similarity = F.normalize(A, dim=-1) @ F.normalize(B, dim=-1).t()
        
        # 幅度编码: [batch_size, n_anchor]
        A_norm = torch.norm(A, dim=-1, keepdim=True)
        B_norm = torch.norm(B, dim=-1, keepdim=True).t()
        magnitude_encoding = torch.log(A_norm + 1e-12) - torch.log(B_norm + 1e-12)
        
        return {
            "direction_similarity": direction_similarity,
            "magnitude_encoding": magnitude_encoding
        }
    
    def _compute_loss(self, converted_embeddings, src_embeddings):
        """
        计算无监督损失：转换后的相对表示应与源相对表示一致
        """
        # 源 embedding 在源 anchor 空间的相对表示
        src_rel = self._embedding2relative(src_embeddings, self.src_anchor_embeddings)
        
        # 转换后的 embedding 在目标 anchor 空间的相对表示
        tgt_rel = self._embedding2relative(converted_embeddings, self.tgt_anchor_embeddings)
        
        # Direction loss: 1 - cosine_similarity
        direction_loss = 1 - F.cosine_similarity(
            tgt_rel['direction_similarity'],
            src_rel['direction_similarity'],
            dim=-1
        ).mean()
        
        # Magnitude loss: Smooth L1
        magnitude_loss = F.smooth_l1_loss(
            tgt_rel['magnitude_encoding'],
            src_rel['magnitude_encoding']
        )
        
        loss = self.lambd * direction_loss + (1 - self.lambd) * magnitude_loss
        return loss, direction_loss, magnitude_loss
    
    def _evaluate(self):
        """在验证集上评估"""
        self.mlp.eval()
        total_loss = 0
        total_dir_loss = 0
        total_mag_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(self.eval_loader, desc="Evaluating", leave=False, ncols=100)
            for (src_batch,) in pbar:
                src_batch = src_batch.to(self.device)
                converted = self.mlp(src_batch)
                loss, dir_loss, mag_loss = self._compute_loss(converted, src_batch)
                
                total_loss += loss.item()
                total_dir_loss += dir_loss.item()
                total_mag_loss += mag_loss.item()
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dir': f'{dir_loss.item():.4f}',
                    'mag': f'{mag_loss.item():.4f}'
                })
        
        n_batches = len(self.eval_loader)
        return total_loss / n_batches, total_dir_loss / n_batches, total_mag_loss / n_batches
    
    def _train(self):
        """训练 MLP"""
        self.logger.info("=" * 80)
        self.logger.info("Training MLPUnsupervisedConverter...")
        self.logger.info(f"  Hidden layers: {self.hidden_layers}")
        self.logger.info(f"  Learning rate: {self.lr}")
        self.logger.info(f"  Epochs: {self.epochs}")
        self.logger.info(f"  Batch size: {self.batch_size}")
        self.logger.info(f"  Lambda (direction weight): {self.lambd}")
        self.logger.info(f"  Eval interval: {self.eval_interval}")
        self.logger.info("=" * 80)
        
        optimizer = torch.optim.AdamW(self.mlp.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        best_eval_loss = float('inf')
        best_state = None
        training_log = []
        
        # 外层 epoch 进度条
        epoch_pbar = tqdm(range(self.epochs), desc="Training", ncols=120)
        
        for epoch in epoch_pbar:
            self.mlp.train()
            total_loss = 0
            total_dir_loss = 0
            total_mag_loss = 0
            
            # 内层 batch 进度条
            batch_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", 
                             leave=False, ncols=100)
            
            for (src_batch,) in batch_pbar:
                src_batch = src_batch.to(self.device)
                
                # 前向传播
                converted = self.mlp(src_batch)
                
                # 计算损失
                loss, dir_loss, mag_loss = self._compute_loss(converted, src_batch)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_dir_loss += dir_loss.item()
                total_mag_loss += mag_loss.item()
                
                # 更新 batch 进度条
                batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dir': f'{dir_loss.item():.4f}',
                    'mag': f'{mag_loss.item():.4f}'
                })
            
            scheduler.step()
            
            n_batches = len(self.train_loader)
            avg_loss = total_loss / n_batches
            avg_dir_loss = total_dir_loss / n_batches
            avg_mag_loss = total_mag_loss / n_batches
            
            # 定期评估
            if (epoch + 1) % self.eval_interval == 0:
                eval_loss, eval_dir_loss, eval_mag_loss = self._evaluate()
                
                log_entry = {
                    'epoch': epoch + 1,
                    'train_loss': avg_loss,
                    'train_dir_loss': avg_dir_loss,
                    'train_mag_loss': avg_mag_loss,
                    'eval_loss': eval_loss,
                    'eval_dir_loss': eval_dir_loss,
                    'eval_mag_loss': eval_mag_loss,
                    'lr': scheduler.get_last_lr()[0]
                }
                training_log.append(log_entry)
                
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.epochs}: "
                    f"train_loss={avg_loss:.6f}, eval_loss={eval_loss:.6f}, "
                    f"train_dir={avg_dir_loss:.6f}, train_mag={avg_mag_loss:.6f}, "
                    f"eval_dir={eval_dir_loss:.6f}, eval_mag={eval_mag_loss:.6f}, "
                    f"lr={scheduler.get_last_lr()[0]:.6f}"
                )
                
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_state = self.mlp.state_dict().copy()
                
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
        }, path)
        self.logger.info(f"Model saved to {path}")
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.mlp.load_state_dict(checkpoint['mlp_state_dict'])
        self.mlp.eval()


if __name__ == "__main__":
    converter = MLPUnsupervisedConverter(
        src_model_path="/data/Llama-3.2-1B-Instruct",
        tgt_model_path="/data/Llama-3.2-3B-Instruct",
        hidden_layers=[8192],
        lr=1e-3,
        epochs=500,
        batch_size=512,
        n_synthetic=5000000,
        eval_ratio=0.01,
        eval_interval=50,
        save_dir="./outputs/mlp_ust_test"
    )
    
    converter.save()
    
    print("\n" + "=" * 80)
    print("Testing conversion...")
    print("=" * 80)
    
    test_tokens = ["hello", "world", "AI"]
    find_similar_tokens(converter, test_tokens)
