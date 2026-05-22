"""
Transfer compression from source model to target model using original dataset.

This module supports multiple transfer methods:
- ls/bp/mlp_ust/random: Use pre-existing converter (from converter module)
- mlp_st: Train a projector with supervised learning  
- e2e: Train a projector end-to-end with reconstruction loss
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import numpy as np
from pathlib import Path
import sys
from typing import List, Dict
from tqdm import tqdm
import logging
from datetime import datetime

# Add paths
sys.path.append(str(Path(__file__).parent.parent / 'simple_compressor'))
sys.path.append(str(Path(__file__).parent.parent / 'converter'))
sys.path.append(str(Path(__file__).parent))

from simple_compressor import SimpleCompressor
from converter_factory import ConverterFactory
from ls_converter import LeastSquaresConverter
from transformers import AutoTokenizer, AutoModelForCausalLM
from projector_mlp import create_projector_mlp
from projector_e2e import create_projector_e2e, E2EWrapper

# Try to import safetensors, fallback if not available
try:
    from safetensors.torch import load_file as safetensors_load
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not available. Will only support PyTorch format.")


def load_dataset(dataset_path: str, num_samples: int = None) -> List[Dict]:
    """Load dataset from JSON file"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to list of dicts with 'text' and 'id' fields
    if isinstance(data, list):
        texts = []
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                text_id = item.get('id', f'text_{idx}')
                text = item.get('text', '')
            else:
                text_id = f'text_{idx}'
                text = str(item)
            texts.append({'id': text_id, 'text': text})
    elif isinstance(data, dict):
        texts = [{'id': k, 'text': v} for k, v in data.items()]
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")
    
    # Limit samples if specified
    if num_samples is not None:
        texts = texts[:num_samples]
    
    return texts


def compress_with_source_model(
    texts: List[Dict],
    src_compressor_path: str,
    src_model_path: str,
    n_mem_tokens: int,
    max_length: int = 512,
    batch_size: int = 32,
    device: str = 'cuda',
) -> List[Dict]:
    """
    Compress texts using source model's trained compressor (batch processing).
    
    Args:
        texts: List of dicts with 'id' and 'text' fields
        src_compressor_path: Path to trained source compressor checkpoint directory
        src_model_path: Source decoder model path (used in training, for converter)
        n_mem_tokens: Number of memory tokens
        max_length: Maximum text length for compression
        batch_size: Batch size for processing (default: 32)
        device: Device to run on
        
    Returns:
        List of compression results with memory vectors
    """
    print(f"Loading source compressor from: {src_compressor_path}")
    print(f"  Decoder model (for converter): {src_model_path}")
    
    compressor = SimpleCompressor.from_pretrained(
        checkpoint_path=src_compressor_path,
        decoder_model_name=src_model_path,
        n_mem_tokens=n_mem_tokens,
        use_lora=False,
        dtype=torch.bfloat16,
        device=device
    )
    
    tokenizer = compressor.compressor_tokenizer
    
    compression_results = []
    
    print(f"Compressing {len(texts)} texts with source model (batch_size={batch_size})...")
    
    # Process in batches
    with torch.no_grad():
        pbar = tqdm(total=len(texts), desc="Compressing", unit="text")
        
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            
            # Extract text IDs and texts
            batch_text_ids = [item['id'] for item in batch_texts]
            batch_texts_str = [item['text'] for item in batch_texts]
            
            # Batch tokenize
            encoded = tokenizer(
                batch_texts_str,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
                padding=True
            )
            
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Batch compress
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                compressed_memories = compressor.compress(input_ids, attention_mask)
            
            # Process each result in the batch
            for i in range(len(batch_text_ids)):
                memory_np = compressed_memories[i].detach().cpu().float().numpy()
                compression_results.append({
                    'text_id': batch_text_ids[i],
                    'memory': memory_np,  # Shape: (n_mem_tokens, src_hidden_dim)
                    'original_text': batch_texts_str[i],
                })
            
            # Update progress bar
            pbar.update(len(batch_texts))
            
            # Clear cache periodically
            del input_ids, attention_mask, compressed_memories
            if (batch_start // batch_size) % 10 == 0:  # Clear cache every 10 batches
                torch.cuda.empty_cache()
        
        pbar.close()
    
    # Cleanup
    del compressor, tokenizer
    torch.cuda.empty_cache()
    
    print(f"✓ Compressed {len(compression_results)} texts successfully")
    
    return compression_results


def transfer_compressed_memories(
    compressed_results: List[Dict],
    src_model_path: str,
    tgt_model_path: str,
    converter_type: str = "ls",
    common_vocab: str = None,
    converter_kwargs: dict = None,
    converter_path: str = None,
    transfer_batch_size: int = 256,
) -> List[Dict]:
    """
    Transfer compressed memories from source dimension to target dimension.
    
    Args:
        compressed_results: Results from compress_with_source_model
        src_model_path: Source model path
        tgt_model_path: Target model path
        converter_type: Converter type (ls, bp, mlp_ust, random, mlp_st)
        common_vocab: Path to common vocabulary file
        converter_kwargs: Additional converter arguments
        converter_path: Path to saved converter model
        transfer_batch_size: Batch size for converter forward pass
        
    Returns:
        Transferred results with converted memory vectors
    """
    print(f"\nInitializing {converter_type} converter for transfer...")
    print(f"  Source: {src_model_path}")
    print(f"  Target: {tgt_model_path}")
    
    factory = ConverterFactory()
    kwargs = converter_kwargs.copy() if converter_kwargs else {}
    
    # Add required parameters
    if 'src_model_path' not in kwargs:
        kwargs['src_model_path'] = src_model_path
    if 'tgt_model_path' not in kwargs:
        kwargs['tgt_model_path'] = tgt_model_path
    if 'common_vocab' not in kwargs and common_vocab is not None:
        kwargs['common_vocab'] = common_vocab
    
    if converter_path:
        print(f"  Loading converter from: {converter_path}")
        kwargs['load_path'] = converter_path
    
    # Create converter
    converter = factory.create_converter(
        converter_type=converter_type,
        **kwargs
    )
    
    # Transfer memories in batches for better throughput
    transferred_results = []
    transfer_batch_size = max(1, int(transfer_batch_size))
    print(f"  Transfer batch size: {transfer_batch_size}")

    for start_idx in tqdm(range(0, len(compressed_results), transfer_batch_size), desc="Transfer batches"):
        batch_results = compressed_results[start_idx:start_idx + transfer_batch_size]
        # Random baseline does not use source memory content, only shape.
        if converter_type == "random":
            first_memory = np.asarray(batch_results[0]['memory'])
            n_mem_tokens = first_memory.shape[0]
            dummy_tensor = torch.empty((len(batch_results), n_mem_tokens, 1), dtype=torch.float32)
            converted_batch = converter.convert(dummy_tensor)
        else:
            batch_memories = [
                np.asarray(result['memory'], dtype=np.float32)
                for result in batch_results
            ]
            # [batch, n_mem_tokens, src_dim] -> [batch, n_mem_tokens, tgt_dim]
            batch_memory_tensor = torch.tensor(np.stack(batch_memories), dtype=torch.float32)
            converted_batch = converter.convert(batch_memory_tensor)
        converted_batch_np = converted_batch.detach().cpu().numpy()

        for result, converted_memory_np in zip(batch_results, converted_batch_np):
            transferred_results.append({
                'text_id': result['text_id'],
                'memory': converted_memory_np.tolist(),
                'original_text': result.get('original_text', ''),
            })
    
    return transferred_results


def train_projector_mlp_st(
    compressed_results: List[Dict],
    src_model_path: str,
    tgt_model_path: str,
    common_vocab: str,
    output_dir: str,
    hidden_layers: List[int] = None,
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 256,
    lambd: float = 0.5,
    dropout: float = 0.1,
    weight_decay: float = 0.01,
    eval_ratio: float = 0.1,
    device: str = 'cuda',
) -> nn.Module:
    """
    Train projector using supervised learning (mlp_st method).
    
    Similar to converter/mlp_st_converter.py but integrated here.
    Uses paired source-target embeddings from common vocab for training.
    
    Args:
        compressed_results: Compressed memory vectors from source model
        src_model_path: Source model path (for getting embeddings)
        tgt_model_path: Target model path (for getting embeddings)
        common_vocab: Path to common vocabulary file
        output_dir: Directory to save trained projector
        hidden_layers: List of hidden layer sizes
        lr: Learning rate
        epochs: Number of training epochs
        batch_size: Training batch size
        lambd: Weight for cosine loss (1-lambd for MSE loss)
        dropout: Dropout rate
        weight_decay: Weight decay for optimizer
        eval_ratio: Ratio of data for evaluation
        device: Device to train on
        
    Returns:
        Trained projector module
    """
    print("\n" + "=" * 60)
    print("Training Projector with Supervised Learning (mlp_st)")
    print("=" * 60)
    
    if hidden_layers is None:
        hidden_layers = [2048]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    log_file = output_dir / f"train_mlp_st_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger('mlp_st_trainer')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info("Loading source and target model embeddings...")
    
    # Get common vocab embeddings
    from base_converter import Converter
    converter_base = Converter(src_model_path, tgt_model_path, common_vocab, "mlp_st")
    src_anchor_embeddings, tgt_anchor_embeddings = converter_base._get_common_embeddings(common_vocab)
    src_anchor_embeddings = src_anchor_embeddings.to(device).to(torch.float32)
    tgt_anchor_embeddings = tgt_anchor_embeddings.to(device).to(torch.float32)
    
    src_dim = src_anchor_embeddings.shape[1]
    tgt_dim = tgt_anchor_embeddings.shape[1]
    
    logger.info(f"Source dimension: {src_dim}")
    logger.info(f"Target dimension: {tgt_dim}")
    logger.info(f"Common vocab pairs: {src_anchor_embeddings.shape[0]}")
    
    # Create projector
    logger.info(f"Creating ProjectorMLP with hidden layers: {hidden_layers}")
    projector = create_projector_mlp(src_dim, tgt_dim, hidden_layers, dropout).to(device)
    
    # Prepare training data
    logger.info("Preparing training data...")
    
    # Add compressed memory vectors to training data
    src_memories = []
    tgt_memories_pseudo = []  # We don't have real target memories, use projector output as pseudo labels
    
    for result in compressed_results:
        memory = np.array(result['memory'])  # (n_mem_tokens, src_dim)
        memory_tensor = torch.tensor(memory, dtype=torch.float32)
        # Flatten: [n_mem_tokens, src_dim] -> n_mem_tokens个 [src_dim]
        for i in range(memory_tensor.shape[0]):
            src_memories.append(memory_tensor[i])
    
    # Combine anchor embeddings and compressed memories
    src_data = torch.stack([src_anchor_embeddings[0]] + src_memories).to(device)
    # For supervised learning, we need target embeddings
    # Use anchor embeddings (we don't have paired compressed vectors)
    tgt_data = torch.stack([tgt_anchor_embeddings[0]]).to(device)
    
    logger.info(f"Training pairs: {src_data.shape[0]}")
    
    # Create dataset and split
    dataset = TensorDataset(src_data, tgt_data.repeat(src_data.shape[0], 1))
    n_total = len(dataset)
    n_eval = max(1, int(n_total * eval_ratio))
    n_train = n_total - n_eval
    
    train_dataset, eval_dataset = random_split(dataset, [n_train, n_eval])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train pairs: {n_train}, Eval pairs: {n_eval}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Lambda (cosine weight): {lambd}")
    logger.info("=" * 60)
    
    best_eval_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        projector.train()
        train_loss = 0
        train_cos_loss = 0
        train_mse_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for src_batch, tgt_batch in pbar:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            # Forward
            converted = projector(src_batch)
            
            # Compute loss
            cosine_loss = 1 - F.cosine_similarity(converted, tgt_batch, dim=-1).mean()
            mse_loss = F.mse_loss(converted, tgt_batch)
            loss = lambd * cosine_loss + (1 - lambd) * mse_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_cos_loss += cosine_loss.item()
            train_mse_loss += mse_loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_cos_loss /= len(train_loader)
        train_mse_loss /= len(train_loader)
        
        # Evaluation
        projector.eval()
        eval_loss = 0
        eval_cos_loss = 0
        eval_mse_loss = 0
        
        with torch.no_grad():
            for src_batch, tgt_batch in eval_loader:
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)
                converted = projector(src_batch)
                
                cosine_loss = 1 - F.cosine_similarity(converted, tgt_batch, dim=-1).mean()
                mse_loss = F.mse_loss(converted, tgt_batch)
                loss = lambd * cosine_loss + (1 - lambd) * mse_loss
                
                eval_loss += loss.item()
                eval_cos_loss += cosine_loss.item()
                eval_mse_loss += mse_loss.item()
        
        eval_loss /= len(eval_loader)
        eval_cos_loss /= len(eval_loader)
        eval_mse_loss /= len(eval_loader)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {train_loss:.4f} (cos: {train_cos_loss:.4f}, mse: {train_mse_loss:.4f}) - "
                   f"Eval Loss: {eval_loss:.4f} (cos: {eval_cos_loss:.4f}, mse: {eval_mse_loss:.4f})")
        
        # Save best model
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            save_path = output_dir / 'best_projector_mlp_st.pt'
            torch.save({
                'projector_state_dict': projector.state_dict(),
                'src_dim': src_dim,
                'tgt_dim': tgt_dim,
                'hidden_layers': hidden_layers,
                'dropout': dropout,
                'epoch': epoch,
                'eval_loss': eval_loss,
            }, save_path)
            logger.info(f"  Saved best model to {save_path}")
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best eval loss: {best_eval_loss:.4f}")
    logger.info("=" * 60)
    
    return projector


def train_projector_e2e(
    compressed_results: List[Dict],
    texts: List[Dict],
    src_model_path: str,
    tgt_model_path: str,
    output_dir: str,
    n_mem_tokens: int,
    src_dim: int,
    tgt_dim: int,
    hidden_layers: List[int] = None,
    lr: float = 1e-4,
    epochs: int = 10,
    batch_size: int = 8,
    max_length: int = 512,
    dropout: float = 0.1,
    weight_decay: float = 0.01,
    common_vocab: str = None,
    init_from_ls: bool = False,
    device: str = 'cuda',
) -> nn.Module:
    """
    Train projector end-to-end with reconstruction loss (e2e method).
    
    Similar to converter/e2e_train.py but integrated here.
    Uses reconstruction loss from target decoder model.
    
    Args:
        compressed_results: Compressed memory vectors from source model
        texts: Original text data
        src_model_path: Source decoder model path (for optional LS init)
        tgt_model_path: Target decoder model path
        output_dir: Directory to save trained projector
        n_mem_tokens: Number of memory tokens
        src_dim: Source dimension
        tgt_dim: Target dimension
        hidden_layers: List of hidden layer sizes
        lr: Learning rate
        epochs: Number of training epochs
        batch_size: Training batch size
        max_length: Maximum text length
        dropout: Dropout rate
        weight_decay: Weight decay for optimizer
        common_vocab: Path to common vocabulary file for LS init
        init_from_ls: Whether to initialize projector from LS mapping
        device: Device to train on
        
    Returns:
        Trained projector module
    """
    print("\n" + "=" * 60)
    print("Training Projector End-to-End (e2e)")
    print("=" * 60)
    
    if hidden_layers is None:
        hidden_layers = [2048]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    log_file = output_dir / f"train_e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger('e2e_trainer')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info("Loading target decoder model...")
    decoder = AutoModelForCausalLM.from_pretrained(tgt_model_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tgt_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Target model: {tgt_model_path}")
    logger.info(f"Source dimension: {src_dim}")
    logger.info(f"Target dimension: {tgt_dim}")
    
    # Create projector
    logger.info(f"Creating ProjectorE2E with hidden layers: {hidden_layers}")
    projector = create_projector_e2e(src_dim, tgt_dim, hidden_layers, dropout).to(device)

    if init_from_ls:
        logger.info("Initializing projector from LS mapping...")
        if hidden_layers and len(hidden_layers) > 0:
            raise ValueError("LS initialization requires hidden_layers=[] for exact linear mapping")

        ls_converter = LeastSquaresConverter(
            src_model_path=src_model_path,
            tgt_model_path=tgt_model_path,
            common_vocab=common_vocab,
            converter_type="ls",
        )
        projection = ls_converter.projection_matrix.detach()  # [src_dim, tgt_dim]

        first_linear = None
        for module in projector.net.modules():
            if isinstance(module, nn.Linear):
                first_linear = module
                break

        if first_linear is None:
            raise RuntimeError("No Linear layer found for LS initialization")
        if first_linear.in_features != src_dim or first_linear.out_features != tgt_dim:
            raise ValueError(
                f"Projector first linear shape mismatch: ({first_linear.in_features}, {first_linear.out_features}) "
                f"vs LS projection ({src_dim}, {tgt_dim})"
            )

        with torch.no_grad():
            first_linear.weight.copy_(projection.T.to(first_linear.weight.dtype).to(first_linear.weight.device))
            if first_linear.bias is not None:
                first_linear.bias.zero_()

        logger.info("LS initialization complete.")
    
    # Create E2E wrapper
    e2e_wrapper = E2EWrapper(projector, decoder, decoder_tokenizer=tokenizer).to(device)
    
    # Prepare dataset
    logger.info("Preparing training dataset...")
    
    # Create mapping from text_id to memory
    text_id_to_memory = {r['text_id']: torch.tensor(r['memory'], dtype=torch.float32) for r in compressed_results}
    
    # Custom dataset
    class E2EDataset(Dataset):
        def __init__(self, texts, text_id_to_memory, tokenizer, max_length):
            self.texts = texts
            self.text_id_to_memory = text_id_to_memory
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text_item = self.texts[idx]
            text_id = text_item['id']
            text = text_item['text']
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt',
                padding='max_length'
            )
            
            # Get memory
            memory = self.text_id_to_memory.get(text_id, torch.zeros(n_mem_tokens, src_dim))
            
            return {
                'memory': memory,
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
            }
    
    dataset = E2EDataset(texts, text_id_to_memory, tokenizer, max_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    logger.info(f"Training samples: {len(dataset)}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {lr}")
    logger.info("=" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        projector.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            memory = batch['memory'].to(device).to(torch.bfloat16)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward
            loss, _ = e2e_wrapper(memory, input_ids, attention_mask)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = output_dir / 'best_projector_e2e.pt'
            torch.save({
                'projector_state_dict': projector.state_dict(),
                'src_dim': src_dim,
                'tgt_dim': tgt_dim,
                'hidden_layers': hidden_layers,
                'dropout': dropout,
                'epoch': epoch,
                'loss': avg_loss,
            }, save_path)
            logger.info(f"  Saved best model to {save_path}")
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info("=" * 60)
    
    # Cleanup
    del decoder, e2e_wrapper
    torch.cuda.empty_cache()
    
    return projector


def transfer_compressor(
    dataset_path: str,
    src_compressor_path: str,
    src_model_path: str,
    tgt_model_path: str,
    output_dir: str,
    n_mem_tokens: int,
    max_length: int = 512,
    num_samples: int = None,
    converter_type: str = "ls",
    common_vocab: str = None,
    converter_kwargs: dict = None,
    converter_path: str = None,
    device: str = 'cuda',
    batch_size: int = 32,
    transfer_batch_size: int = 256,
) -> dict:
    """
    Main transfer pipeline: dataset -> source compression -> converter -> target format.
    
    Args:
        dataset_path: Path to original dataset JSON
        src_compressor_path: Path to trained source compressor checkpoint
        src_model_path: Source model path
        tgt_model_path: Target model path
        output_dir: Directory to save transferred results
        n_mem_tokens: Number of memory tokens
        max_length: Maximum text length for compression
        num_samples: Limit number of samples (None = all)
        converter_type: Converter type (ls, bp, mlp_ust, random, mlp_st, e2e, ls_e2e)
        common_vocab: Path to common vocabulary file
        converter_kwargs: Additional converter arguments
        converter_path: Path to saved converter model
        device: Device to run on
        batch_size: Batch size for compression (default: 32)
        transfer_batch_size: Batch size for converter transfer step (default: 256)
        
    Returns:
        Output data dict
    """
    # Step 1: Load dataset
    print("=" * 60)
    print("Step 1: Loading dataset")
    print("=" * 60)
    texts = load_dataset(dataset_path, num_samples)
    print(f"Loaded {len(texts)} texts from {dataset_path}")
    
    # Step 2: Compress with source model
    print("\n" + "=" * 60)
    print("Step 2: Compressing with source model")
    print("=" * 60)
    compressed_results = compress_with_source_model(
        texts=texts,
        src_compressor_path=src_compressor_path,
        src_model_path=src_model_path,
        n_mem_tokens=n_mem_tokens,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )
    
    # Step 3: Transfer to target model dimension
    print("\n" + "=" * 60)
    print("Step 3: Transferring to target model")
    print("=" * 60)
    
    # Check if using trainable projector methods
    if converter_type in ['mlp_st', 'e2e', 'ls_e2e']:
        # Get dimensions
        src_dim = compressed_results[0]['memory'].shape[-1] if compressed_results else None
        if src_dim is None:
            raise ValueError("Cannot determine source dimension from compression results")
        
        # Load target model to get dimension
        from transformers import AutoConfig
        tgt_config = AutoConfig.from_pretrained(tgt_model_path)
        tgt_dim = tgt_config.hidden_size
        
        # Train projector
        if converter_type == 'mlp_st':
            print("  Using supervised learning (mlp_st)")
            if common_vocab is None:
                raise ValueError("common_vocab is required for mlp_st method")
            
            projector = train_projector_mlp_st(
                compressed_results=compressed_results,
                src_model_path=src_model_path,
                tgt_model_path=tgt_model_path,
                common_vocab=common_vocab,
                output_dir=output_dir,
                hidden_layers=converter_kwargs.get('hidden_layers', None) if converter_kwargs else None,
                lr=converter_kwargs.get('lr', 1e-3) if converter_kwargs else 1e-3,
                epochs=converter_kwargs.get('epochs', 100) if converter_kwargs else 100,
                batch_size=converter_kwargs.get('batch_size', 256) if converter_kwargs else 256,
                lambd=converter_kwargs.get('lambd', 0.5) if converter_kwargs else 0.5,
                dropout=converter_kwargs.get('dropout', 0.1) if converter_kwargs else 0.1,
                weight_decay=converter_kwargs.get('weight_decay', 0.01) if converter_kwargs else 0.01,
                device=device,
            )
        else:  # e2e / ls_e2e
            if converter_type == 'ls_e2e':
                print("  Using hybrid training (LS init + short e2e fine-tune)")
            else:
                print("  Using end-to-end training (e2e)")
            e2e_hidden_layers = converter_kwargs.get('hidden_layers', None) if converter_kwargs else None
            if converter_type == 'ls_e2e' and e2e_hidden_layers is None:
                e2e_hidden_layers = []
            projector = train_projector_e2e(
                compressed_results=compressed_results,
                texts=texts,
                src_model_path=src_model_path,
                tgt_model_path=tgt_model_path,
                output_dir=output_dir,
                n_mem_tokens=n_mem_tokens,
                src_dim=src_dim,
                tgt_dim=tgt_dim,
                hidden_layers=e2e_hidden_layers,
                lr=converter_kwargs.get('lr', 1e-4) if converter_kwargs else 1e-4,
                epochs=converter_kwargs.get('epochs', 2 if converter_type == 'ls_e2e' else 10) if converter_kwargs else (2 if converter_type == 'ls_e2e' else 10),
                batch_size=converter_kwargs.get('batch_size', 8) if converter_kwargs else 8,
                max_length=max_length,
                dropout=converter_kwargs.get('dropout', 0.1) if converter_kwargs else 0.1,
                weight_decay=converter_kwargs.get('weight_decay', 0.01) if converter_kwargs else 0.01,
                common_vocab=common_vocab,
                init_from_ls=(converter_type == 'ls_e2e'),
                device=device,
            )
        
        # Apply trained projector to all memories
        print(f"\nApplying trained projector to {len(compressed_results)} memories...")
        transferred_results = []
        projector.eval()
        with torch.no_grad():
            for result in tqdm(compressed_results, desc="Applying projector"):
                memory = np.array(result['memory'])
                memory_tensor = torch.tensor(memory, dtype=torch.float32).to(device)
                converted_memory = projector(memory_tensor)
                converted_memory_np = converted_memory.detach().cpu().numpy()
                
                transferred_results.append({
                    'text_id': result['text_id'],
                    'memory': converted_memory_np.tolist(),
                    'original_text': result.get('original_text', ''),
                })
    else:
        # Use existing converter methods (ls, bp, mlp_ust, random, etc.)
        print(f"  Using converter method: {converter_type}")
        transferred_results = transfer_compressed_memories(
            compressed_results=compressed_results,
            src_model_path=src_model_path,
            tgt_model_path=tgt_model_path,
            converter_type=converter_type,
            common_vocab=common_vocab,
            converter_kwargs=converter_kwargs,
            converter_path=converter_path,
            transfer_batch_size=transfer_batch_size,
        )
    
    # Step 4: Save results
    print("\n" + "=" * 60)
    print("Step 4: Saving results")
    print("=" * 60)
    
    # Extract experiment name from source compressor path
    src_exp_name = Path(src_compressor_path).parent.name
    
    # Infer decoder model used during training from checkpoint path
    # Path format: {compressor}_to_{decoder}_mem{n}_len{l}_ds_4gpu
    decoder_model_name = None
    if '_to_' in src_exp_name:
        parts = src_exp_name.split('_to_')
        decoder_name = parts[1].split('_mem')[0]  # Extract decoder name before _mem
        
        # Map model names to paths
        model_paths = {
            'gpt2': '${MODELS_DIR}/gpt2',
            'llama1b': '${MODELS_DIR}/Llama-3.2-1B-Instruct',
            'llama3b': '${MODELS_DIR}/Llama-3.2-3B-Instruct',
            'llama8b': '${MODELS_DIR}/Llama-3-8B-Instruct',
            'qwen1.5b': '${MODELS_DIR}/Qwen/Qwen2.5-1.5B-Instruct',
            'qwen3b': '${MODELS_DIR}/Qwen/Qwen2.5-3B-Instruct',
            'qwen7b': '${MODELS_DIR}/Qwen/Qwen2.5-7B-Instruct',
        }
        
        if decoder_name in model_paths:
            decoder_model_name = model_paths[decoder_name]
    
    output_data = {
        'metadata': {
            'dataset_path': dataset_path,
            'source_compressor': src_compressor_path,
            'source_model': src_model_path,  # Used for converter
            'decoder_model': decoder_model_name,  # Decoder model used during training (for origin evaluation)
            'target_model': tgt_model_path,
            'source_experiment': src_exp_name,
            'n_mem_tokens': n_mem_tokens,
            'max_length': max_length,
            'num_texts': len(transferred_results),
            'transfer_method': converter_type,
        },
        'results': transferred_results
    }
    
    # Save to output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = f"compression_results_from_{src_exp_name}_using_{converter_type}.json"
    output_path = output_dir / output_filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nTransferred results saved to: {output_path}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(
        description='Transfer compression from source model to target model using original dataset'
    )
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to original dataset JSON')
    parser.add_argument('--src_compressor', type=str, required=True,
                       help='Path to trained source compressor checkpoint')
    parser.add_argument('--src_model', type=str, required=True,
                       help='Source model path')
    parser.add_argument('--tgt_model', type=str, required=True,
                       help='Target model path')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save transferred results')
    parser.add_argument('--n_mem_tokens', type=int, required=True,
                       help='Number of memory tokens')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum text length for compression')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Limit number of samples (None = all)')
    parser.add_argument('--converter_type', type=str, default='ls',
                       choices=['ls', 'bp', 'mlp_ust', 'random', 'mlp_st', 'e2e', 'ls_e2e'],
                       help='Transfer method: ls/bp/mlp_ust/random (use converter), mlp_st (supervised train), e2e (end-to-end train), ls_e2e (LS init + short e2e)')
    parser.add_argument('--common_vocab', type=str, default=None,
                       help='Path to common vocabulary file (required for mlp_st)')
    parser.add_argument('--converter_kwargs', type=str, default='{}',
                       help='Additional converter kwargs as JSON string (e.g., {"hidden_layers":[2048],"epochs":100})')
    parser.add_argument('--converter_path', type=str, default=None,
                       help='Path to saved converter model (for ls/bp/mlp_ust methods)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for compression (default: 32)')
    parser.add_argument('--transfer_batch_size', type=int, default=256,
                       help='Batch size for transfer conversion (default: 256)')
    
    args = parser.parse_args()
    converter_kwargs = json.loads(args.converter_kwargs)
    
    print("=" * 60)
    print("Compressor Transfer Pipeline")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Source compressor: {args.src_compressor}")
    print(f"Source model: {args.src_model}")
    print(f"Target model: {args.tgt_model}")
    print(f"Memory tokens: {args.n_mem_tokens}")
    print(f"Transfer method: {args.converter_type}")
    print(f"Compression batch size: {args.batch_size}")
    print(f"Transfer batch size: {args.transfer_batch_size}")
    if args.converter_type == 'mlp_st':
        print(f"  -> Supervised learning with common vocab")
    elif args.converter_type == 'e2e':
        print(f"  -> End-to-end training with reconstruction loss")
    elif args.converter_type == 'ls_e2e':
        print(f"  -> Hybrid: LS initialization + short end-to-end fine-tuning")
    else:
        print(f"  -> Using pre-existing converter")
    print("=" * 60)
    
    transfer_compressor(
        dataset_path=args.dataset,
        src_compressor_path=args.src_compressor,
        src_model_path=args.src_model,
        tgt_model_path=args.tgt_model,
        output_dir=args.output_dir,
        n_mem_tokens=args.n_mem_tokens,
        max_length=args.max_length,
        num_samples=args.num_samples,
        converter_type=args.converter_type,
        common_vocab=args.common_vocab,
        converter_kwargs=converter_kwargs,
        converter_path=args.converter_path,
        device=args.device,
        batch_size=args.batch_size,
        transfer_batch_size=args.transfer_batch_size,
    )
    
    print("\n" + "=" * 60)
    print("Transfer Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
