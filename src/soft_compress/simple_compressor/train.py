'''
Training script for simple compressor
'''

import torch
import torch.distributed as dist
import json
import os
from pathlib import Path
from transformers import HfArgumentParser, set_seed
from datasets import Dataset
from argument import TrainArguments, DataArguments
from simple_compressor import SimpleCompressor
from datacollator import CompressorDataCollator
from base_trainer import CompressorTrainer


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
    Process batch by batch to avoid OOM (instead of collecting all predictions)
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
            
            # Model's labels structure: [prefix (-100)] + [target (decoder_input_ids)]
            # labels = [prefix (1+n_mem_tokens个-100)] + [target (seq_len个token)]
            # Logits shape: (batch_size, 1+n_mem_tokens+seq_len, vocab_size)
            # After shift: shift_logits = logits[:, :-1, :] predicts shift_labels = labels[:, 1:]
            # shift_labels = [prefix的后n_mem_tokens个-100] + [target的所有seq_len个token]
            
            # Shift for causal LM: logits[:, :-1, :] predicts labels[:, 1:]
            shift_logits = logits[:, :-1, :]  # (batch_size, n_mem_tokens+seq_len, vocab_size)
            
            # Build target labels: prepare decoder_input_ids with padding masked
            target_labels = decoder_input_ids.clone()
            target_labels[decoder_attention_mask == 0] = -100  # mask padding
            
            # Reconstruct shift_labels to match shift_logits exactly
            # shift_labels should have shape (batch_size, shift_logits.shape[1])
            # Structure: [prefix的后n_mem_tokens个-100] + [target的所有seq_len个token]
            shift_labels_len = shift_logits.shape[1]  # Should be n_mem_tokens + seq_len
            seq_len = decoder_input_ids.shape[1]
            target_slice_len = shift_labels_len - n_mem_tokens  # Length available for target
            
            shift_labels = torch.full(
                (decoder_input_ids.shape[0], shift_labels_len),
                -100,
                dtype=decoder_input_ids.dtype,
                device=decoder_input_ids.device
            )
            
            # Fill target part: use the appropriate slice of target_labels
            # shift_labels[:, n_mem_tokens:] should match the target portion
            if target_slice_len <= seq_len:
                # Take first target_slice_len tokens from target_labels
                shift_labels[:, n_mem_tokens:] = target_labels[:, :target_slice_len]
            else:
                # Pad if needed (shouldn't happen normally, but handle gracefully)
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
            
            # Calculate accuracy only for valid tokens (labels != -100)
            mask = (shift_labels != -100)
            correct = (preds == shift_labels) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
            # Clear cache to free memory
            del outputs, logits, decoder_input_ids, decoder_attention_mask, shift_logits, shift_labels, target_labels, preds, mask, correct
            torch.cuda.empty_cache()
    
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    return accuracy


def main():
    # Parse arguments
    parser = HfArgumentParser((TrainArguments, DataArguments))
    train_args, data_args = parser.parse_args_into_dataclasses()
    
    # Set seed
    set_seed(train_args.random_seed)
    
    # Disable wandb
    train_args.report_to = []
    
    # Check if using DeepSpeed
    use_deepspeed = (
        train_args.deepspeed is not None or 
        hasattr(train_args, 'deepspeed_config_file') and train_args.deepspeed_config_file is not None or
        os.environ.get('DEEPSPEED_CONFIG_FILE') is not None
    )
    
    # Set device (only used when not using DeepSpeed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if train_args.bf16 else (torch.float16 if train_args.fp16 else torch.float32)
    
    # Check if this is the main process (for multi-GPU training)
    # When using DeepSpeed, check LOCAL_RANK or RANK environment variable
    if use_deepspeed:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        rank = int(os.environ.get('RANK', 0))
        is_main_process = (local_rank == 0) or (rank == 0)
    else:
        # For single GPU or non-DeepSpeed multi-GPU, always print
        is_main_process = True
    
    if is_main_process:
        print("=" * 60)
        print("Simple Compressor Training")
        print("=" * 60)
        print(f"Compressor model: {train_args.compress_model}")
        print(f"Decoder model: {train_args.decoder_model}")
        print(f"Memory tokens: {train_args.embed_len}")
        print(f"Segment length: {train_args.segment_length}")
        print(f"Use LoRA: {train_args.use_lora}")
        print(f"Using DeepSpeed: {use_deepspeed}")
        if use_deepspeed:
            deepspeed_config = train_args.deepspeed or os.environ.get('DEEPSPEED_CONFIG_FILE', 'N/A')
            print(f"DeepSpeed config: {deepspeed_config}")
        else:
            print(f"Device: {device}")
        print(f"Dtype: {dtype}")
        print("=" * 60)
    
    # Load datasets (all ranks must load to ensure synchronization)
    if is_main_process:
        print("\nLoading datasets...")
    train_dataset = load_dataset(data_args.train_data_dir, data_args.train_data_samples)
    eval_dataset = load_dataset(data_args.valid_data_dir, data_args.valid_data_samples)
    if is_main_process:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Eval samples: {len(eval_dataset)}")
    
    # Ensure all ranks have loaded datasets before proceeding
    if use_deepspeed and dist.is_initialized():
        dist.barrier()
        if is_main_process:
            print("All ranks synchronized after dataset loading")
    
    # Initialize model (all ranks must initialize)
    if is_main_process:
        print("\nInitializing model...")
    
    # Ensure synchronization before model initialization
    if use_deepspeed and dist.is_initialized():
        dist.barrier()
    
    model = SimpleCompressor(
        compressor_model_name=train_args.compress_model,
        decoder_model_name=train_args.decoder_model,
        n_mem_tokens=train_args.embed_len,
        use_lora=train_args.use_lora,
        lora_config=None,  # Use default config
        dtype=dtype
    )
    
    # Only move model to device if not using DeepSpeed
    # DeepSpeed will handle device placement automatically
    if not use_deepspeed:
        model.to(device)
    
    # Ensure synchronization after model initialization
    if use_deepspeed and dist.is_initialized():
        dist.barrier()
        if is_main_process:
            print("Model initialized on all ranks")
    
    # Print trainable parameters (only on main process)
    if is_main_process:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Data collator
    data_collator = CompressorDataCollator(
        compressor_tokenizer=model.compressor_tokenizer,
        decoder_tokenizer=model.decoder_tokenizer,
        max_length=train_args.segment_length
    )
    
    # Initialize trainer
    trainer = CompressorTrainer(
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
    
    # Save model (only on main process when using DeepSpeed)
    if is_main_process:
        print("\nSaving model...")
    trainer.save_model()
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
    
    # Save metrics (only on main process)
    if is_main_process:
        metrics_file = Path(train_args.output_dir) / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump({
                'train_loss': train_result.training_loss,
                'eval_loss': eval_result['eval_loss'],
                'accuracy': accuracy
            }, f, indent=2)


if __name__ == "__main__":
    main()

