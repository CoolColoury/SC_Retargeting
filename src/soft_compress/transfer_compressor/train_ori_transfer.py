"""
Training script for OriTransfer Compressor

Direct continuation training from source compressor to target decoder:
- Example: gpt2-to-llama1b -> gpt2-to-llama8b
- Uses OriTransferCompressor (no additional projector)
- Supports two training modes:
  1. converter_only: Only train Converter (freeze Encoder)
  2. encoder_converter: Train both Encoder and Converter
- Standard Trainer-based training
"""

import torch
import torch.distributed as dist
import json
import os
from pathlib import Path
from transformers import HfArgumentParser, set_seed, TrainingArguments, Trainer
from datasets import Dataset
from dataclasses import dataclass, field
import sys

# Add paths
sys.path.append(str(Path(__file__).parent.parent / 'simple_compressor'))
sys.path.append(str(Path(__file__).parent))

from ori_transfer_compressor_model import OriTransferCompressor
from datacollator import CompressorDataCollator  # Reuse from simple_compressor


@dataclass
class OriTransferTrainArguments(TrainingArguments):
    """Training arguments for OriTransfer compressor"""
    
    # Model paths
    src_compressor_path: str = field(
        default=None,
        metadata={"help": "Path to trained source compressor checkpoint"}
    )
    src_decoder_model_path: str = field(
        default=None,
        metadata={"help": "Source decoder model path (for loading source compressor)"}
    )
    tgt_model_path: str = field(
        default=None,
        metadata={"help": "Target model path"}
    )
    
    # Model config
    embed_len: int = field(
        default=32,
        metadata={"help": "Number of memory tokens"}
    )
    segment_length: int = field(
        default=256,
        metadata={"help": "Maximum text length"}
    )
    train_mode: str = field(
        default="converter_only",
        metadata={"help": "Training mode: 'converter_only' or 'encoder_converter'"}
    )
    
    # Random seed
    random_seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    
    # Checkpoint resuming
    resume_from_checkpoint: bool = field(
        default=False,
        metadata={"help": "Whether to resume from checkpoint"}
    )
    last_ckpt_dir: str = field(
        default=None,
        metadata={"help": "Path to last checkpoint"}
    )


@dataclass
class DataArguments:
    """Data arguments"""
    train_data_dir: str = field(
        default=None,
        metadata={"help": "Path to training data JSON file"}
    )
    valid_data_dir: str = field(
        default=None,
        metadata={"help": "Path to validation data JSON file"}
    )
    train_data_samples: int = field(
        default=None,
        metadata={"help": "Limit number of training samples"}
    )
    valid_data_samples: int = field(
        default=None,
        metadata={"help": "Limit number of validation samples"}
    )


def load_dataset(data_path, num_samples=None):
    """
    Load dataset from JSON file
    Expected format: list of dicts with 'text' field or dict with text values
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to list of dicts with 'text' field
    if isinstance(data, list):
        texts = [{'text': item['text'] if isinstance(item, dict) else item} for item in data]
    elif isinstance(data, dict):
        texts = [{'text': v} for v in data.values()]
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")
    
    # Limit samples if specified
    if num_samples is not None:
        texts = texts[:num_samples]
    
    return Dataset.from_list(texts)


def calculate_accuracy(trainer, eval_dataset):
    """
    Calculate reconstruction accuracy on evaluation set
    Process batch by batch to avoid OOM
    """
    trainer.model.eval()
    eval_dataloader = trainer.get_eval_dataloader()
    n_mem_tokens = trainer.model.n_mem_tokens
    
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in eval_dataloader:
            # Move batch to device
            batch = trainer._prepare_inputs(batch)
            
            # Get model outputs
            outputs = trainer.model(
                compress_input_ids=batch['compress_input_ids'],
                compress_attention_mask=batch['compress_attention_mask'],
                decoder_input_ids=batch['decoder_input_ids'],
                decoder_attention_mask=batch['decoder_attention_mask']
            )
            
            logits = outputs.logits
            decoder_input_ids = batch['decoder_input_ids']
            decoder_attention_mask = batch['decoder_attention_mask']
            
            # Shift for causal LM: logits[:, :-1, :] predicts labels[:, 1:]
            shift_logits = logits[:, :-1, :]
            
            # Build target labels
            target_labels = decoder_input_ids.clone()
            target_labels[decoder_attention_mask == 0] = -100
            
            # Reconstruct shift_labels to match shift_logits
            shift_labels_len = shift_logits.shape[1]
            seq_len = decoder_input_ids.shape[1]
            target_slice_len = shift_labels_len - n_mem_tokens
            
            shift_labels = torch.full(
                (decoder_input_ids.shape[0], shift_labels_len),
                -100,
                dtype=decoder_input_ids.dtype,
                device=decoder_input_ids.device
            )
            
            if target_slice_len <= seq_len:
                shift_labels[:, n_mem_tokens:] = target_labels[:, :target_slice_len]
            else:
                padded = torch.full(
                    (decoder_input_ids.shape[0], target_slice_len),
                    -100,
                    dtype=decoder_input_ids.dtype,
                    device=decoder_input_ids.device
                )
                padded[:, :seq_len] = target_labels
                shift_labels[:, n_mem_tokens:] = padded
            
            # Get predictions
            preds = torch.argmax(shift_logits, dim=-1)
            
            # Calculate accuracy
            mask = (shift_labels != -100)
            correct = (preds == shift_labels) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
            # Clear cache
            del outputs, logits, decoder_input_ids, decoder_attention_mask, shift_logits, shift_labels, target_labels, preds, mask, correct
            torch.cuda.empty_cache()
    
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    return accuracy


class OriTransferCompressorTrainer(torch.nn.Module):
    """
    Wrapper for OriTransferCompressor to work with Trainer
    Handles the forward pass and loss computation
    """
    def __init__(self, ori_transfer_compressor):
        super().__init__()
        self.model = ori_transfer_compressor
        self.n_mem_tokens = ori_transfer_compressor.n_mem_tokens
    
    def forward(self, compress_input_ids, compress_attention_mask,
                decoder_input_ids, decoder_attention_mask):
        """Forward pass for training"""
        return self.model(
            compress_input_ids=compress_input_ids,
            compress_attention_mask=compress_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask
        )


class OriTransferCompressorTrainerClass(Trainer):
    """
    Custom Trainer for OriTransferCompressor
    Handles saving with proper model structure
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss for OriTransfer compressor training
        """
        outputs = model(
            compress_input_ids=inputs['compress_input_ids'],
            compress_attention_mask=inputs['compress_attention_mask'],
            decoder_input_ids=inputs['decoder_input_ids'],
            decoder_attention_mask=inputs['decoder_attention_mask']
        )
        
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Prediction step for evaluation
        """
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        
        if prediction_loss_only:
            return (loss, None, None)
        
        logits = outputs.logits
        labels = inputs['decoder_input_ids']
        
        return (loss, logits, labels)
    
    def _save(self, output_dir=None, state_dict=None):
        """
        Save model with handling of shared weights
        Adapted for OriTransferCompressor structure
        """
        model = self.model
        
        # Get the actual OriTransferCompressor from the wrapper
        ori_transfer_compressor = model.model
        
        # Store original references for restoration
        compressor_lm_head_orig = None
        decoder_lm_head_orig = None
        
        # Handle compressor shared weights
        if hasattr(ori_transfer_compressor, 'compressor'):
            compressor = ori_transfer_compressor.compressor
            if hasattr(compressor, 'lm_head') and hasattr(compressor, 'model'):
                if hasattr(compressor.model, 'embed_tokens'):
                    compressor_lm_head = compressor.lm_head.weight
                    compressor_embed_tokens = compressor.model.embed_tokens.weight
                    if compressor_lm_head.data_ptr() == compressor_embed_tokens.data_ptr():
                        compressor_lm_head_orig = compressor_lm_head
                        compressor.lm_head.weight = torch.nn.Parameter(compressor_lm_head.clone())
        
        # Handle decoder shared weights
        if hasattr(ori_transfer_compressor, 'decoder'):
            decoder = ori_transfer_compressor.decoder
            if hasattr(decoder, 'lm_head') and hasattr(decoder, 'model'):
                if hasattr(decoder.model, 'embed_tokens'):
                    decoder_lm_head = decoder.lm_head.weight
                    decoder_embed_tokens = decoder.model.embed_tokens.weight
                    if decoder_lm_head.data_ptr() == decoder_embed_tokens.data_ptr():
                        decoder_lm_head_orig = decoder_lm_head
                        decoder.lm_head.weight = torch.nn.Parameter(decoder_lm_head.clone())
        
        # Temporarily disable safetensors to avoid shared weight issues
        original_use_safetensors = getattr(self.args, 'use_safetensors', None)
        self.args.use_safetensors = False
        
        # Save using parent method
        super()._save(output_dir, state_dict)
        
        # Restore original setting
        if original_use_safetensors is not None:
            self.args.use_safetensors = original_use_safetensors
        
        # Restore original shared weights
        if compressor_lm_head_orig is not None:
            ori_transfer_compressor.compressor.lm_head.weight = compressor_lm_head_orig
        
        if decoder_lm_head_orig is not None:
            ori_transfer_compressor.decoder.lm_head.weight = decoder_lm_head_orig


def main():
    # Parse arguments
    parser = HfArgumentParser((OriTransferTrainArguments, DataArguments))
    train_args, data_args = parser.parse_args_into_dataclasses()
    
    # Set seed
    set_seed(train_args.random_seed)
    
    # Disable wandb
    train_args.report_to = []
    
    # Disable column removal - data collator handles tokenization
    train_args.remove_unused_columns = False
    
    # Check if using DeepSpeed
    use_deepspeed = (
        train_args.deepspeed is not None or 
        hasattr(train_args, 'deepspeed_config_file') and train_args.deepspeed_config_file is not None or
        os.environ.get('DEEPSPEED_CONFIG_FILE') is not None
    )
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if train_args.bf16 else (torch.float16 if train_args.fp16 else torch.float32)
    
    # Check if main process
    if use_deepspeed:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        rank = int(os.environ.get('RANK', 0))
        is_main_process = (local_rank == 0) or (rank == 0)
    else:
        is_main_process = True
    
    if is_main_process:
        print("=" * 60)
        print("OriTransfer Compressor Training")
        print("=" * 60)
        print(f"Source compressor: {train_args.src_compressor_path}")
        print(f"Source decoder: {train_args.src_decoder_model_path}")
        print(f"Target model: {train_args.tgt_model_path}")
        print(f"Memory tokens: {train_args.embed_len}")
        print(f"Segment length: {train_args.segment_length}")
        print(f"Training mode: {train_args.train_mode}")
        print(f"Using DeepSpeed: {use_deepspeed}")
        if use_deepspeed:
            deepspeed_config = train_args.deepspeed or os.environ.get('DEEPSPEED_CONFIG_FILE', 'N/A')
            print(f"DeepSpeed config: {deepspeed_config}")
        else:
            print(f"Device: {device}")
        print(f"Dtype: {dtype}")
        print("=" * 60)
    
    # Load datasets
    if is_main_process:
        print("\nLoading datasets...")
    train_dataset = load_dataset(data_args.train_data_dir, data_args.train_data_samples)
    eval_dataset = load_dataset(data_args.valid_data_dir, data_args.valid_data_samples)
    if is_main_process:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Eval samples: {len(eval_dataset)}")
    
    # Synchronize
    if use_deepspeed and dist.is_initialized():
        dist.barrier()
        if is_main_process:
            print("All ranks synchronized after dataset loading")
    
    # Initialize model
    if is_main_process:
        print("\nInitializing OriTransfer Compressor...")
    
    if use_deepspeed and dist.is_initialized():
        dist.barrier()
    
    ori_transfer_compressor = OriTransferCompressor(
        src_compressor_path=train_args.src_compressor_path,
        src_decoder_model_path=train_args.src_decoder_model_path,
        tgt_model_path=train_args.tgt_model_path,
        n_mem_tokens=train_args.embed_len,
        train_mode=train_args.train_mode,
        dtype=dtype
    )
    
    # Wrap for training
    model = OriTransferCompressorTrainer(ori_transfer_compressor)
    
    # Move to device if not using DeepSpeed
    if not use_deepspeed:
        model.to(device)
    
    # Synchronize
    if use_deepspeed and dist.is_initialized():
        dist.barrier()
        if is_main_process:
            print("Model initialized on all ranks")
    
    # Print trainable parameters
    if is_main_process:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Data collator - reuse from simple compressor
    # Get tokenizers from ori_transfer compressor
    data_collator = CompressorDataCollator(
        compressor_tokenizer=ori_transfer_compressor.compressor_tokenizer,
        decoder_tokenizer=ori_transfer_compressor.decoder_tokenizer,
        max_length=train_args.segment_length
    )
    
    # Initialize trainer with custom OriTransferCompressorTrainerClass
    trainer = OriTransferCompressorTrainerClass(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Resume from checkpoint if specified
    checkpoint = None
    if train_args.resume_from_checkpoint and train_args.last_ckpt_dir:
        checkpoint = train_args.last_ckpt_dir
        if is_main_process:
            print(f"\nResuming from checkpoint: {checkpoint}")
    
    # Train
    if is_main_process:
        print("\nStarting training...")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save model
    if is_main_process:
        print("\nSaving model...")
    
    # Save ori_transfer compressor
    save_path = Path(train_args.output_dir)
    ori_transfer_compressor.save_pretrained(save_path)
    
    trainer.save_state()
    
    # Evaluate
    if is_main_process:
        print("\nEvaluating...")
    eval_result = trainer.evaluate()
    accuracy = calculate_accuracy(trainer, eval_dataset)
    eval_result['accuracy'] = accuracy
    
    if is_main_process:
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Final eval loss: {eval_result['eval_loss']:.4f}")
        print(f"Final accuracy: {accuracy:.4f}")
        print(f"Model saved to: {train_args.output_dir}")
        print("=" * 60)
    
    # Save metrics
    if is_main_process:
        metrics_file = Path(train_args.output_dir) / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump({
                'train_loss': train_result.training_loss,
                'eval_loss': eval_result['eval_loss'],
                'accuracy': accuracy,
                'train_mode': train_args.train_mode
            }, f, indent=2)


if __name__ == "__main__":
    main()
