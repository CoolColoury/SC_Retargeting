"""
Evaluation script for Simple Compressor model.
Tests text reconstruction quality using the same metrics as simple_mem.

This script evaluates:
1. Reconstruction loss
2. Token-level accuracy 
3. Text generation quality (BLEU, ROUGE, BERTScore)
4. Generated text comparison
"""

import argparse
import json
import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from transformers import AutoTokenizer
from safetensors.torch import load_file as safetensors_load
import sys

# Add simple_compressor to path
sys.path.append(str(Path(__file__).parent.parent / 'simple_compressor'))
from simple_compressor import SimpleCompressor

# BLEU and ROUGE (shared implementation; smoothed sentence BLEU)
try:
    from .text_generation_scores import cal_bleu_rouge

    HAS_BLEU_ROUGE = True
except ImportError:
    try:
        from text_generation_scores import cal_bleu_rouge

        HAS_BLEU_ROUGE = True
    except ImportError:
        print("Warning: nltk or rouge-score not installed. BLEU/ROUGE metrics will be skipped.")
        print("Install with: pip install nltk rouge-score")

        def cal_bleu_rouge(pred: str, ref: str) -> Dict:
            return {
                "bleu": 0.0,
                "bleu1": 0.0,
                "bleu2": 0.0,
                "bleu3": 0.0,
                "bleu4": 0.0,
                "rougeL": 0.0,
            }

        HAS_BLEU_ROUGE = False


def load_test_dataset(dataset_path: str, max_samples: int = None) -> List[Dict]:
    """Load test dataset"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    return data


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> Dict:
    """
    Calculate token-level accuracy.
    Same implementation as simple_mem/train.py
    
    Args:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len)
    
    Returns:
        Dictionary with accuracy metrics
    """
    # Shift logits and labels for causal LM
    # logits[:, :-1, :] predicts labels[:, 1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # Get predictions
    predictions = torch.argmax(shift_logits, dim=-1)
    
    # Create mask for valid positions (ignore -100)
    mask = (shift_labels != -100)
    
    # Calculate accuracy
    correct = (predictions == shift_labels) & mask
    total_correct = correct.sum().item()
    total_tokens = mask.sum().item()
    
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    
    # Per-token accuracy
    per_token_correct = correct.float()
    per_token_mask = mask.float()
    
    return {
        'accuracy': accuracy,
        'total_correct': total_correct,
        'total_tokens': total_tokens,
        'per_token_correct': per_token_correct,
        'per_token_mask': per_token_mask
    }


def evaluate_reconstruction(
    model: SimpleCompressor,
    compress_input_ids: torch.Tensor,
    compress_attention_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_attention_mask: torch.Tensor,
    device: str = 'cuda',
    generated_ids: List[int] = None,  # Optional: pre-generated IDs to avoid re-generation
    max_new_tokens: int = 512,
    generate_text: bool = True,
    temperature: float = 0.0,
) -> Dict:
    """
    Evaluate reconstruction quality using autoregressive generation (not teacher forcing).
    This matches the actual generation process and provides more realistic metrics.
    
    Returns:
        Dictionary with loss and accuracy metrics
    """
    model.eval()
    
    with torch.no_grad():
        # Calculate loss and token accuracy using teacher forcing.
        # Keep this consistent with transfer_compressor evaluation logic.
        outputs = model(
            compress_input_ids=compress_input_ids.to(device),
            compress_attention_mask=compress_attention_mask.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            decoder_attention_mask=decoder_attention_mask.to(device)
        )
        loss = outputs.loss.item()
        # Match decoder forward layout: [BOS] + [MEMORY] + [TARGET]
        # Prefix positions are ignored in loss/accuracy via -100.
        batch_size = decoder_input_ids.size(0)
        prefix_labels = torch.full(
            (batch_size, model.n_mem_tokens + 1),
            -100,
            dtype=decoder_input_ids.dtype,
            device=device
        )
        target_labels = decoder_input_ids.to(device).clone()
        target_labels[decoder_attention_mask.to(device) == 0] = -100
        labels = torch.cat([prefix_labels, target_labels], dim=1)
        acc_metrics = calculate_accuracy(outputs.logits, labels)
        accuracy = acc_metrics['accuracy']
        total_correct = acc_metrics['total_correct']
        total_tokens = acc_metrics['total_tokens']
    
    return {
        'loss': loss,  # Note: This is teacher-forcing loss, not generation loss
        'accuracy': accuracy,  # This is generation-based accuracy (autoregressive)
        'total_correct': total_correct,
        'total_tokens': total_tokens
    }


def generate_reconstructed_text(
    model: SimpleCompressor,
    compress_input_ids: torch.Tensor,
    compress_attention_mask: torch.Tensor,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    device: str = 'cuda'
) -> List[int]:
    """
    Generate reconstructed text from compressed input.
    
    Args:
        model: SimpleCompressor model
        compress_input_ids: Input to compress
        compress_attention_mask: Attention mask for input
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        device: Device
    
    Returns:
        Generated token IDs
    """
    model.eval()

    def _normalize_token_id(token_id):
        if token_id is None:
            return None
        if isinstance(token_id, (list, tuple)):
            if len(token_id) == 0:
                return None
            return int(token_id[0])
        return int(token_id)

    with torch.inference_mode():
        compressed_memory = model.compress(
            compress_input_ids.to(device),
            compress_attention_mask.to(device)
        )

        # Must match SimpleCompressor.forward: tokenizer bos, else tokenizer eos.
        # Qwen2.5 sets config.bos_token_id to pad (151643) while tokenizer.bos is None;
        # forward() uses eos (151645). Using config.bos here breaks generation vs training.
        bos_token_id = _normalize_token_id(model.decoder_tokenizer.bos_token_id)
        if bos_token_id is None:
            bos_token_id = _normalize_token_id(model.decoder_tokenizer.eos_token_id)
        eos_token_id = _normalize_token_id(model.decoder.config.eos_token_id)
        if eos_token_id is None:
            eos_token_id = _normalize_token_id(model.decoder_tokenizer.eos_token_id)

        decoder_embeddings = model.decoder.get_input_embeddings()
        bos_embedding = decoder_embeddings(
            torch.tensor([[bos_token_id]], device=device)
        )
        prefix_embeds = torch.cat([bos_embedding, compressed_memory], dim=1)
        prefix_len = prefix_embeds.shape[1]

        prefix_attention_mask = torch.ones(
            (1, prefix_len),
            dtype=torch.long,
            device=device
        )

        generate_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(temperature and float(temperature) > 0.0),
            "temperature": float(temperature) if temperature and float(temperature) > 0.0 else None,
            "pad_token_id": eos_token_id,
            "eos_token_id": eos_token_id,
            "use_cache": True,
            "return_dict_in_generate": False,
        }
        generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            sequences = model.decoder.generate(
                inputs_embeds=prefix_embeds,
                attention_mask=prefix_attention_mask,
                **generate_kwargs
            )

        seq = sequences[0].tolist()
        if len(seq) > prefix_len:
            return seq[prefix_len:]
        return seq


def compute_text_similarity_metrics(
    original_text: str,
    generated_text: str
) -> Dict:
    """
    Compute text similarity metrics (BLEU, ROUGE, exact match, etc.)
    
    Args:
        original_text: Original text
        generated_text: Generated text
    
    Returns:
        Dictionary with similarity metrics
    """
    import difflib
    from collections import Counter
    
    # Exact match
    exact_match = (original_text.strip() == generated_text.strip())
    
    # Character-level similarity
    char_similarity = difflib.SequenceMatcher(
        None, original_text, generated_text
    ).ratio()
    
    # Word-level metrics
    original_words = original_text.lower().split()
    generated_words = generated_text.lower().split()
    
    # Word overlap (unigram precision/recall/F1)
    original_counter = Counter(original_words)
    generated_counter = Counter(generated_words)
    
    overlap = sum((original_counter & generated_counter).values())
    
    precision = overlap / len(generated_words) if len(generated_words) > 0 else 0
    recall = overlap / len(original_words) if len(original_words) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # BLEU and ROUGE scores
    bleu_rouge_metrics = cal_bleu_rouge(generated_text, original_text)
    
    result = {
        'exact_match': exact_match,
        'char_similarity': char_similarity,
        'word_precision': precision,
        'word_recall': recall,
        'word_f1': f1,
        'original_length': len(original_text),
        'generated_length': len(generated_text),
        'original_words': len(original_words),
        'generated_words': len(generated_words)
    }
    
    # Add BLEU and ROUGE metrics
    result.update(bleu_rouge_metrics)
    
    return result


def evaluate_model(
    model_path: str,
    compressor_model_name: str,
    decoder_model_name: str,
    n_mem_tokens: int,
    test_data_path: str,
    output_path: str = None,
    max_samples: int = None,
    max_length: int = 512,
    device: str = 'cuda',
    generate_text: bool = True,
    generate_samples: int = 50,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    use_lora: bool = False,
):
    """
    Evaluate a trained SimpleCompressor model.
    
    Args:
        model_path: Path to trained model checkpoint
        compressor_model_name: Base compressor model name
        decoder_model_name: Base decoder model name
        n_mem_tokens: Number of memory tokens
        test_data_path: Path to test dataset
        output_path: Path to save results
        max_samples: Maximum samples to evaluate
        max_length: Maximum sequence length
        device: Device to use
        generate_text: Whether to generate text
        max_new_tokens: Max tokens to generate
        temperature: Generation temperature
        use_lora: Whether model uses LoRA
    """
    print("="*80)
    print("Simple Compressor Evaluation")
    print("="*80)
    print(f"Model path: {model_path}")
    print(f"Compressor: {compressor_model_name}")
    print(f"Decoder: {decoder_model_name}")
    print(f"Memory tokens: {n_mem_tokens}")
    print(f"Test data: {test_data_path}")
    print(f"Device: {device}")
    print("="*80)
    print()
    
    # Load model
    print("Loading model...")
    model = SimpleCompressor(
        compressor_model_name=compressor_model_name,
        decoder_model_name=decoder_model_name,
        n_mem_tokens=n_mem_tokens,
        use_lora=use_lora,
        dtype=torch.bfloat16
    )
    
    # Load trained weights - support both safetensors and pytorch_model.bin
    model_path_obj = Path(model_path)
    checkpoint_path_safetensors = model_path_obj / 'model.safetensors'
    checkpoint_path_pytorch = model_path_obj / 'pytorch_model.bin'
    
    if checkpoint_path_safetensors.exists():
        print(f"Loading from safetensors: {checkpoint_path_safetensors}")
        checkpoint = safetensors_load(checkpoint_path_safetensors)
        model.load_state_dict(checkpoint, strict=False)
    elif checkpoint_path_pytorch.exists():
        print(f"Loading from pytorch_model.bin: {checkpoint_path_pytorch}")
        checkpoint = torch.load(checkpoint_path_pytorch, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
    else:
        raise FileNotFoundError(
            f"Model checkpoint not found. Expected either:\n"
            f"  - {checkpoint_path_safetensors}\n"
            f"  - {checkpoint_path_pytorch}"
        )
    
    model.to(device)
    
    # Ensure Projector weights are in bfloat16 to match the model dtype
    # This is needed because Projector might have been saved in float32
    if hasattr(model, 'projector'):
        for param in model.projector.parameters():
            if param.dtype != torch.bfloat16:
                param.data = param.data.to(torch.bfloat16)
    
    model.eval()
    
    print("✓ Model loaded successfully")
    print()
    
    # Load test data
    print("Loading test data...")
    test_data = load_test_dataset(test_data_path, max_samples)
    print(f"✓ Loaded {len(test_data)} test samples")
    print()
    
    # Prepare tokenizers
    compressor_tokenizer = model.compressor_tokenizer
    decoder_tokenizer = model.decoder_tokenizer
    
    # Evaluation results
    all_results = []
    total_loss = 0
    total_accuracy = 0
    total_correct = 0
    total_tokens = 0
    
    # BLEU and ROUGE lists for aggregation
    bleu_list, bleu1_list, bleu2_list = [], [], []
    bleu3_list, bleu4_list, rougeL_list = [], [], []
    
    # Evaluate each sample
    print("Evaluating samples...")
    print(f"Note: Autoregressive generation can be slow (~{max_length} forward passes per sample)")
    if generate_text:
        if int(generate_samples) > 0:
            print(f"Detailed generation metrics on first {int(generate_samples)} samples")
        else:
            print("Detailed generation metrics on all samples")
    print()
    for idx, sample in enumerate(tqdm(test_data, desc="Evaluating")):
        text = sample['text']
        text_id = sample.get('id', f'sample_{idx}')
        
        # Tokenize for compressor
        compress_inputs = compressor_tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors='pt'
        )
        
        # Tokenize for decoder (target)
        decoder_inputs = decoder_tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors='pt'
        )
        
        # Generate text first (if requested) to avoid duplicate generation
        generated_ids = None
        should_generate_this = bool(generate_text) and (int(generate_samples) == 0 or idx < int(generate_samples))
        if should_generate_this:
            # Respect the configured generation budget for fair evaluation.
            generated_ids = generate_reconstructed_text(
                model=model,
                compress_input_ids=compress_inputs['input_ids'],
                compress_attention_mask=compress_inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device
            )
        
        # Evaluate reconstruction (reuse generated_ids if available to avoid duplicate generation)
        metrics = evaluate_reconstruction(
            model=model,
            compress_input_ids=compress_inputs['input_ids'],
            compress_attention_mask=compress_inputs['attention_mask'],
            decoder_input_ids=decoder_inputs['input_ids'],
            decoder_attention_mask=decoder_inputs['attention_mask'],
            device=device,
            generated_ids=generated_ids,  # Pass pre-generated IDs to avoid re-generation
            max_new_tokens=max_new_tokens,
            generate_text=should_generate_this,
            temperature=temperature,
        )
        
        result = {
            'text_id': text_id,
            'text': text,
            'loss': metrics['loss'],
            'accuracy': metrics['accuracy'],
            'total_correct': metrics['total_correct'],
            'total_tokens': metrics['total_tokens'],
            'compress_length': compress_inputs['input_ids'].size(1),
            'target_length': decoder_inputs['input_ids'].size(1),
        }
        
        # Use generated text if available
        if should_generate_this and generated_ids is not None:
            
            generated_text = decoder_tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Truncate original text to match generated text length (in tokens)
            # This ensures fair comparison for BLEU score
            generated_token_count = len(generated_ids)
            original_token_ids = decoder_inputs['input_ids'][0].cpu().tolist()
            
            # Remove padding tokens
            original_mask = decoder_inputs['attention_mask'][0].cpu().tolist()
            original_token_ids_filtered = [tok for tok, mask in zip(original_token_ids, original_mask) if mask == 1]
            
            # Remove BOS and EOS if present
            bos_token_id = decoder_tokenizer.bos_token_id
            eos_token_id = decoder_tokenizer.eos_token_id
            if original_token_ids_filtered and original_token_ids_filtered[0] == bos_token_id:
                original_token_ids_filtered = original_token_ids_filtered[1:]
            if original_token_ids_filtered and original_token_ids_filtered[-1] == eos_token_id:
                original_token_ids_filtered = original_token_ids_filtered[:-1]
            
            # Truncate original to match generated length
            original_token_ids_truncated = original_token_ids_filtered[:generated_token_count]
            original_text_truncated = decoder_tokenizer.decode(original_token_ids_truncated, skip_special_tokens=True)
            
            # Compute text similarity using truncated original text
            similarity_metrics = compute_text_similarity_metrics(original_text_truncated, generated_text)
            
            # Also store original text length info for reference
            similarity_metrics['original_text_truncated'] = original_text_truncated
            similarity_metrics['original_token_count'] = len(original_token_ids_filtered)
            similarity_metrics['generated_token_count'] = generated_token_count
            
            result['generated_text'] = generated_text
            result['generated_length'] = len(generated_ids)
            result.update(similarity_metrics)
            
            # Accumulate BLEU and ROUGE for summary
            if HAS_BLEU_ROUGE:
                bleu_list.append(similarity_metrics['bleu'])
                bleu1_list.append(similarity_metrics['bleu1'])
                bleu2_list.append(similarity_metrics['bleu2'])
                bleu3_list.append(similarity_metrics['bleu3'])
                bleu4_list.append(similarity_metrics['bleu4'])
                rougeL_list.append(similarity_metrics['rougeL'])
        
        all_results.append(result)
        
        # Accumulate stats
        total_loss += metrics['loss']
        total_accuracy += metrics['accuracy']
        total_correct += metrics['total_correct']
        total_tokens += metrics['total_tokens']
    
    # Compute summary statistics
    num_samples = len(all_results)
    avg_loss = total_loss / num_samples
    avg_accuracy = total_accuracy / num_samples
    overall_accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    summary = {
        'model_path': model_path,
        'compressor_model': compressor_model_name,
        'decoder_model': decoder_model_name,
        'n_mem_tokens': n_mem_tokens,
        'num_samples': num_samples,
        'avg_loss': avg_loss,
        'avg_accuracy': avg_accuracy,
        'overall_accuracy': overall_accuracy,
        'total_correct_tokens': total_correct,
        'total_tokens': total_tokens,
    }
    
    generated_results = [r for r in all_results if 'generated_text' in r]
    if generate_text and len(generated_results) > 0:
        # Compute average text similarity metrics
        denom = len(generated_results)
        avg_exact_match = sum(r.get('exact_match', 0) for r in generated_results) / denom
        avg_char_sim = sum(r.get('char_similarity', 0) for r in generated_results) / denom
        avg_word_f1 = sum(r.get('word_f1', 0) for r in generated_results) / denom
        
        summary['avg_exact_match'] = avg_exact_match
        summary['avg_char_similarity'] = avg_char_sim
        summary['avg_word_f1'] = avg_word_f1
        
        # Add BLEU and ROUGE averages
        if HAS_BLEU_ROUGE and len(bleu_list) > 0:
            summary['avg_bleu'] = float(np.mean(bleu_list))
            summary['avg_bleu1'] = float(np.mean(bleu1_list))
            summary['avg_bleu2'] = float(np.mean(bleu2_list))
            summary['avg_bleu3'] = float(np.mean(bleu3_list))
            summary['avg_bleu4'] = float(np.mean(bleu4_list))
            summary['avg_rougeL'] = float(np.mean(rougeL_list))
        else:
            summary['avg_bleu'] = 0.0
            summary['avg_bleu1'] = 0.0
            summary['avg_bleu2'] = 0.0
            summary['avg_bleu3'] = 0.0
            summary['avg_bleu4'] = 0.0
            summary['avg_rougeL'] = 0.0
    
    # Print summary
    print()
    print("="*80)
    print("Evaluation Summary")
    print("="*80)
    print(f"Samples evaluated: {num_samples}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Average accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    print(f"Overall accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"Total tokens: {total_tokens:,} | Correct: {total_correct:,}")
    
    if generate_text:
        print(f"\nText Generation Metrics:")
        print(f"  Exact match rate: {summary['avg_exact_match']:.4f} ({summary['avg_exact_match']*100:.2f}%)")
        print(f"  Avg char similarity: {summary['avg_char_similarity']:.4f}")
        print(f"  Avg word F1: {summary['avg_word_f1']:.4f}")
        if HAS_BLEU_ROUGE:
            print(f"  Avg BLEU: {summary['avg_bleu']:.4f}")
            print(f"  Avg BLEU-1: {summary['avg_bleu1']:.4f}")
            print(f"  Avg BLEU-2: {summary['avg_bleu2']:.4f}")
            print(f"  Avg BLEU-3: {summary['avg_bleu3']:.4f}")
            print(f"  Avg BLEU-4: {summary['avg_bleu4']:.4f}")
            print(f"  Avg ROUGE-L: {summary['avg_rougeL']:.4f}")
        else:
            print(f"  Note: BLEU/ROUGE metrics not available (install nltk and rouge-score)")
    
    print("="*80)
    
    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            'summary': summary,
            'results': all_results
        }
        
        # Atomic write: avoids leaving a truncated JSON if the process dies mid-dump.
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_path, output_path)
        # Same-process sanity check (detects torn writes / FS issues before returning).
        json.loads(output_path.read_text(encoding="utf-8"))

        print(f"\n✓ Results saved to: {output_path}")
    
    return summary, all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Simple Compressor model')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--compressor_model', type=str, required=True,
                       help='Base compressor model name')
    parser.add_argument('--decoder_model', type=str, required=True,
                       help='Base decoder model name')
    parser.add_argument('--n_mem_tokens', type=int, default=None,
                       help='Number of memory tokens (if not provided, will infer from checkpoint path)')
    parser.add_argument('--use_lora', action='store_true',
                       help='Whether model uses LoRA')
    
    # Data arguments
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test dataset JSON')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to evaluate')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    
    # Evaluation arguments
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--no_generate', action='store_true',
                       help='Skip text generation')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                       help='Maximum tokens to generate')
    parser.add_argument('--generate_samples', type=int, default=50,
                       help='Only first N samples run detailed text generation (0 = all)')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Generation temperature (0 = greedy)')
    
    args = parser.parse_args()
    
    # Infer n_mem_tokens from checkpoint path if not provided
    n_mem_tokens = args.n_mem_tokens
    if n_mem_tokens is None:
        model_path_obj = Path(args.model_path)
        dir_name = model_path_obj.name
        if '_mem' in dir_name:
            try:
                # Extract number after _mem
                mem_part = dir_name.split('_mem')[1].split('_')[0]
                n_mem_tokens = int(mem_part)
                print(f"Inferred n_mem_tokens from checkpoint path: {n_mem_tokens}")
            except (ValueError, IndexError):
                raise ValueError(
                    f"Could not infer n_mem_tokens from checkpoint path: {args.model_path}\n"
                    f"Please provide --n_mem_tokens explicitly."
                )
        else:
            raise ValueError(
                f"Could not infer n_mem_tokens from checkpoint path: {args.model_path}\n"
                f"Please provide --n_mem_tokens explicitly."
            )
    
    # Run evaluation
    evaluate_model(
        model_path=args.model_path,
        compressor_model_name=args.compressor_model,
        decoder_model_name=args.decoder_model,
        n_mem_tokens=n_mem_tokens,
        test_data_path=args.test_data,
        output_path=args.output_path,
        max_samples=args.max_samples,
        max_length=args.max_length,
        device=args.device,
        generate_text=not args.no_generate,
        generate_samples=args.generate_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_lora=args.use_lora,
    )


if __name__ == '__main__':
    main()

