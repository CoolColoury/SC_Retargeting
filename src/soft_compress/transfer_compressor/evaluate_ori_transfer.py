"""
Evaluation script for OriTransferCompressor model.
Uses the same evaluation logic as SimpleCompressor (test_simple_compressor.py).
"""

import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from transformers.utils import logging as hf_logging

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

try:
    from soft_compress.evaluation.text_generation_scores import cal_bleu_rouge

    HAS_BLEU_ROUGE = True
except ImportError:
    print("Warning: nltk or rouge-score not installed. BLEU/ROUGE metrics will be skipped.")
    HAS_BLEU_ROUGE = False

    def cal_bleu_rouge(pred: str, ref: str) -> Dict:
        return {
            "bleu": 0.0,
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "rougeL": 0.0,
        }

from ori_transfer_compressor_model import OriTransferCompressor
from decoder_prefix_tokens import decoder_bos_token_id, decoder_eos_token_id


def load_test_dataset(dataset_path: str, max_samples: int = None) -> List[Dict]:
    """Load test dataset"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    return data


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> Dict:
    """
    Calculate token-level accuracy using teacher forcing.
    Same implementation as simple_compressor/train.py
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
    
    return {
        'accuracy': accuracy,
        'total_correct': total_correct,
        'total_tokens': total_tokens
    }


def evaluate_reconstruction_batch(
    model: OriTransferCompressor,
    compress_input_ids: torch.Tensor,
    compress_attention_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_attention_mask: torch.Tensor,
    device: str = 'cuda'
) -> Dict:
    """
    Evaluate reconstruction quality for a batch of samples.
    Uses teacher forcing for both loss and accuracy (consistent with training).
    Returns per-sample metrics.
    """
    model.eval()

    with torch.inference_mode():
        outputs = model(
            compress_input_ids=compress_input_ids.to(device),
            compress_attention_mask=compress_attention_mask.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            decoder_attention_mask=decoder_attention_mask.to(device)
        )

        batch_size = decoder_input_ids.size(0)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        prefix_labels = torch.full(
            (batch_size, model.n_mem_tokens + 1),
            -100,
            dtype=decoder_input_ids.dtype,
            device=device
        )
        target_labels = decoder_input_ids.to(device).clone()
        target_labels[decoder_attention_mask.to(device) == 0] = -100
        labels = torch.cat([prefix_labels, target_labels], dim=1)

        shift_logits = logits[:, :-1, :].contiguous()   # (B, L, V)
        shift_labels = labels[:, 1:].contiguous()       # (B, L)

        mask = (shift_labels != -100)
        token_counts = mask.sum(dim=1)  # (B,)

        # Accuracy per sample
        predictions = torch.argmax(shift_logits, dim=-1)
        correct = ((predictions == shift_labels) & mask).sum(dim=1)  # (B,)

        # Loss per sample (NLL on masked tokens), vectorized
        log_probs = torch.log_softmax(shift_logits, dim=-1)  # (B, L, V)
        safe_labels = shift_labels.clamp_min(0)
        token_logp = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, L)
        nll = (-token_logp) * mask.to(token_logp.dtype)  # (B, L)
        loss_sum = nll.sum(dim=1)  # (B,)
        loss_mean = torch.where(token_counts > 0, loss_sum / token_counts.to(loss_sum.dtype), torch.zeros_like(loss_sum))

        results = []
        for i in range(batch_size):
            tokens_i = int(token_counts[i].item())
            correct_i = int(correct[i].item())
            loss_i = float(loss_mean[i].item()) if tokens_i > 0 else 0.0
            acc_i = (correct_i / tokens_i) if tokens_i > 0 else 0.0
            results.append({
                'loss': loss_i,
                'accuracy': acc_i,
                'total_correct': correct_i,
                'total_tokens': tokens_i
            })

        return results


def evaluate_reconstruction(
    model: OriTransferCompressor,
    compress_input_ids: torch.Tensor,
    compress_attention_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_attention_mask: torch.Tensor,
    device: str = 'cuda',
    generated_ids: List[int] = None
) -> Dict:
    """
    Evaluate reconstruction quality.
    Uses teacher forcing for both loss and accuracy (consistent with training).
    """
    model.eval()
    
    with torch.no_grad():
        # Calculate loss and accuracy using teacher forcing
        outputs = model(
            compress_input_ids=compress_input_ids.to(device),
            compress_attention_mask=compress_attention_mask.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            decoder_attention_mask=decoder_attention_mask.to(device)
        )
        
        loss = outputs.loss.item()
        
        # Calculate accuracy from logits and labels
        # The model's forward method returns logits, but we need to extract labels
        # Labels are constructed inside the model's forward, so we need to reconstruct them
        # Actually, we can use the decoder_input_ids as labels (with proper masking)
        
        # Get logits from outputs
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)
        
        # Prepare labels: [-100] * (1 + n_mem) + decoder_input_ids
        batch_size = decoder_input_ids.size(0)
        prefix_labels = torch.full(
            (batch_size, model.n_mem_tokens + 1),
            -100,
            dtype=decoder_input_ids.dtype,
            device=device
        )
        target_labels = decoder_input_ids.clone().to(device)
        target_labels[decoder_attention_mask.to(device) == 0] = -100
        labels = torch.cat([prefix_labels, target_labels], dim=1)
        
        # Calculate accuracy
        accuracy_metrics = calculate_accuracy(logits, labels)
        accuracy = accuracy_metrics['accuracy']
        total_correct = accuracy_metrics['total_correct']
        total_tokens = accuracy_metrics['total_tokens']
        
        # Generate text for BLEU/ROUGE (only if not provided and needed)
        # Note: generated_ids is only used for BLEU/ROUGE, not for accuracy calculation
        if generated_ids is None:
            target_length = decoder_input_ids.size(1)
            generated_ids = generate_reconstructed_text(
                model=model,
                compress_input_ids=compress_input_ids,
                compress_attention_mask=compress_attention_mask,
                max_new_tokens=target_length,
                temperature=0.0,
                device=device
            )
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'total_correct': total_correct,
        'total_tokens': total_tokens,
        'generated_ids': generated_ids  # For BLEU/ROUGE calculation (optional)
    }


def generate_reconstructed_text(
    model: OriTransferCompressor,
    compress_input_ids: torch.Tensor,
    compress_attention_mask: torch.Tensor,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    device: str = 'cuda'
) -> List[int]:
    """
    Generate reconstructed text.
    Same logic as SimpleCompressor generation.
    """
    model.eval()
    
    with torch.inference_mode():
        # Compress input -> memory in target decoder space
        compressed_memory = model.compress(
            compress_input_ids.to(device),
            compress_attention_mask.to(device)
        )

        # Match SimpleCompressor.forward / test_simple_compressor.generate_reconstructed_text
        bos_token_id = decoder_bos_token_id(model.decoder_tokenizer, model.decoder.config)
        eos_token_id = decoder_eos_token_id(model.decoder_tokenizer, model.decoder.config)

        # Prefix embeds: [BOS] + [Memory]
        decoder_embeddings = model.decoder.get_input_embeddings()
        bos_embedding = decoder_embeddings(
            torch.tensor([[bos_token_id]], device=device)
        )
        prefix_embeds = torch.cat([bos_embedding, compressed_memory], dim=1)  # [1, 1+n_mem, dim]
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
            # Avoid repeated warnings: explicitly set pad_token_id for open-end generation.
            "pad_token_id": eos_token_id,
            "eos_token_id": eos_token_id,
            "use_cache": True,
            "return_dict_in_generate": False,
        }
        # Remove None values for compatibility
        generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

        # Use HF generate() to leverage KV-cache (avoid O(T^2) full recompute loop)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            sequences = model.decoder.generate(
                inputs_embeds=prefix_embeds,
                attention_mask=prefix_attention_mask,
                **generate_kwargs
            )

        # With inputs_embeds, HF may return sequences that include dummy prefix token ids.
        # We only need the newly generated token ids.
        seq = sequences[0].tolist()
        if len(seq) > prefix_len:
            gen = seq[prefix_len:]
        else:
            gen = seq
        return gen


def compute_text_similarity_metrics(original_text: str, generated_text: str) -> Dict:
    """Compute text similarity metrics"""
    import difflib
    from collections import Counter
    
    exact_match = (original_text.strip() == generated_text.strip())
    
    char_similarity = difflib.SequenceMatcher(
        None, original_text, generated_text
    ).ratio()
    
    original_words = original_text.lower().split()
    generated_words = generated_text.lower().split()
    
    original_counter = Counter(original_words)
    generated_counter = Counter(generated_words)
    
    overlap = sum((original_counter & generated_counter).values())
    
    precision = overlap / len(generated_words) if len(generated_words) > 0 else 0
    recall = overlap / len(original_words) if len(original_words) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
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
    
    result.update(bleu_rouge_metrics)
    
    return result


def evaluate_model(
    compressor_checkpoint: str,
    src_compressor_path: str,
    src_decoder_model_path: str,
    tgt_model_path: str,
    n_mem_tokens: int,
    train_mode: str,
    test_data_path: str,
    output_path: str = None,
    max_samples: int = None,
    max_length: int = 512,
    device: str = 'cuda',
    generate_text: bool = True,
    generate_samples: int = 0,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    batch_size: int = 1,
):
    """
    Evaluate OriTransferCompressor model.
    Same logic as test_simple_compressor.py
    """
    print("="*80)
    print("OriTransfer Compressor Evaluation")
    print("="*80)
    print(f"Checkpoint: {compressor_checkpoint}")
    print(f"Train mode: {train_mode}")
    print(f"Memory tokens: {n_mem_tokens}")
    print(f"Test data: {test_data_path}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"Generate text: {generate_text}")
    if generate_text and generate_samples:
        print(f"Generate samples: {generate_samples} (only first N samples will generate)")
    print("="*80)
    print()
    
    # Load model
    print("Loading model...")
    # Reduce HF generate() informational spam in logs
    hf_logging.set_verbosity_error()
    model = OriTransferCompressor.from_pretrained(
        checkpoint_path=compressor_checkpoint,
        src_compressor_path=src_compressor_path,
        src_decoder_model_path=src_decoder_model_path,
        tgt_model_path=tgt_model_path,
        n_mem_tokens=n_mem_tokens,
        train_mode=train_mode,
        dtype=torch.bfloat16
    )
    
    model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully")
    print()
    
    # Load test data
    print("Loading test data...")
    test_data = load_test_dataset(test_data_path, max_samples)
    print(f"✓ Loaded {len(test_data)} test samples")
    print()
    
    # Prepare tokenizers (re-use those inside OriTransferCompressor)
    compressor_tokenizer = model.compressor_tokenizer
    decoder_tokenizer = model.decoder_tokenizer
    
    # Evaluation results
    all_results = []
    total_loss = 0
    total_accuracy = 0
    total_correct = 0
    total_tokens = 0
    
    bleu_list, bleu1_list, bleu2_list = [], [], []
    bleu3_list, bleu4_list, rougeL_list = [], [], []
    
    # Evaluate samples in batches
    print(f"Evaluating samples with batch_size={batch_size}...")
    num_batches = (len(test_data) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Evaluating batches"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(test_data))
        batch_samples = test_data[batch_start:batch_end]
        
        # Prepare batch data
        batch_texts = [sample['text'] for sample in batch_samples]
        batch_ids = [sample.get('id', f'sample_{batch_start + i}') for i, sample in enumerate(batch_samples)]
        
        # Tokenize batch with padding
        compress_inputs = compressor_tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        
        decoder_inputs = decoder_tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        
        # Evaluate reconstruction in batch (teacher forcing for loss & accuracy)
        batch_metrics = evaluate_reconstruction_batch(
            model=model,
            compress_input_ids=compress_inputs['input_ids'],
            compress_attention_mask=compress_inputs['attention_mask'],
            decoder_input_ids=decoder_inputs['input_ids'],
            decoder_attention_mask=decoder_inputs['attention_mask'],
            device=device
        )
        
        # Process each sample in the batch
        for i, sample in enumerate(batch_samples):
            text = sample['text']
            text_id = batch_ids[i]
            metrics = batch_metrics[i]
        
            result = {
                'text_id': text_id,
                'text': text,
                'loss': metrics['loss'],
                'accuracy': metrics['accuracy'],
                'total_correct': metrics['total_correct'],
                'total_tokens': metrics['total_tokens'],
                'compress_length': (compress_inputs['attention_mask'][i].sum().item()),
                'target_length': (decoder_inputs['attention_mask'][i].sum().item()),
            }
            
            # Generate text (expensive). Optionally only do it for first N samples overall.
            do_generate = generate_text
            if generate_text and generate_samples and len(all_results) >= generate_samples:
                do_generate = False

            if do_generate:
                # Get single sample inputs
                single_compress_input_ids = compress_inputs['input_ids'][i:i+1]
                single_compress_mask = compress_inputs['attention_mask'][i:i+1]
                single_decoder_input_ids = decoder_inputs['input_ids'][i:i+1]
                single_decoder_mask = decoder_inputs['attention_mask'][i:i+1]
                
                # Trim padding from single sample
                compress_valid_len = single_compress_mask.sum().item()
                single_compress_input_ids = single_compress_input_ids[:, :compress_valid_len]
                single_compress_mask = single_compress_mask[:, :compress_valid_len]
                
                target_length = single_decoder_mask.sum().item()
                
                generated_ids = generate_reconstructed_text(
                    model=model,
                    compress_input_ids=single_compress_input_ids,
                    compress_attention_mask=single_compress_mask,
                    max_new_tokens=target_length,
                    temperature=temperature,
                    device=device
                )
                
                generated_text = decoder_tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Truncate original text to match generated length
                generated_token_count = len(generated_ids)
                original_token_ids = single_decoder_input_ids[0].cpu().tolist()
                original_mask = single_decoder_mask[0].cpu().tolist()
                original_token_ids_filtered = [tok for tok, mask in zip(original_token_ids, original_mask) if mask == 1]
                
                bos_token_id = decoder_tokenizer.bos_token_id
                eos_token_id = decoder_tokenizer.eos_token_id
                if original_token_ids_filtered and original_token_ids_filtered[0] == bos_token_id:
                    original_token_ids_filtered = original_token_ids_filtered[1:]
                if original_token_ids_filtered and original_token_ids_filtered[-1] == eos_token_id:
                    original_token_ids_filtered = original_token_ids_filtered[:-1]
                
                original_token_ids_truncated = original_token_ids_filtered[:generated_token_count]
                original_text_truncated = decoder_tokenizer.decode(original_token_ids_truncated, skip_special_tokens=True)
                
                similarity_metrics = compute_text_similarity_metrics(original_text_truncated, generated_text)
                
                similarity_metrics['original_text_truncated'] = original_text_truncated
                similarity_metrics['original_token_count'] = len(original_token_ids_filtered)
                similarity_metrics['generated_token_count'] = generated_token_count
                
                result['generated_text'] = generated_text
                result['generated_length'] = len(generated_ids)
                result.update(similarity_metrics)
                
                if HAS_BLEU_ROUGE:
                    bleu_list.append(similarity_metrics['bleu'])
                    bleu1_list.append(similarity_metrics['bleu1'])
                    bleu2_list.append(similarity_metrics['bleu2'])
                    bleu3_list.append(similarity_metrics['bleu3'])
                    bleu4_list.append(similarity_metrics['bleu4'])
                    rougeL_list.append(similarity_metrics['rougeL'])
            
            all_results.append(result)

            # Accumulate summary stats across ALL samples (not last sample only)
            total_loss += metrics['loss']
            total_accuracy += metrics['accuracy']
            total_correct += metrics['total_correct']
            total_tokens += metrics['total_tokens']
    
    # Compute summary
    num_samples = len(all_results)
    avg_loss = total_loss / num_samples
    avg_accuracy = total_accuracy / num_samples
    overall_accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    summary = {
        'compressor_checkpoint': compressor_checkpoint,
        'src_compressor': src_compressor_path,
        'tgt_model': tgt_model_path,
        'train_mode': train_mode,
        'n_mem_tokens': n_mem_tokens,
        'num_samples': num_samples,
        'avg_loss': avg_loss,
        'avg_accuracy': avg_accuracy,
        'overall_accuracy': overall_accuracy,
        'total_correct_tokens': total_correct,
        'total_tokens': total_tokens,
    }
    
    if generate_text and len(all_results) > 0:
        avg_exact_match = sum(r.get('exact_match', 0) for r in all_results) / num_samples
        avg_char_sim = sum(r.get('char_similarity', 0) for r in all_results) / num_samples
        avg_word_f1 = sum(r.get('word_f1', 0) for r in all_results) / num_samples
        
        summary['avg_exact_match'] = avg_exact_match
        summary['avg_char_similarity'] = avg_char_sim
        summary['avg_word_f1'] = avg_word_f1
        
        if HAS_BLEU_ROUGE and len(bleu_list) > 0:
            summary['avg_bleu'] = float(np.mean(bleu_list))
            summary['avg_bleu1'] = float(np.mean(bleu1_list))
            summary['avg_bleu2'] = float(np.mean(bleu2_list))
            summary['avg_bleu3'] = float(np.mean(bleu3_list))
            summary['avg_bleu4'] = float(np.mean(bleu4_list))
            summary['avg_rougeL'] = float(np.mean(rougeL_list))
    
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
    
    print("="*80)
    
    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            'summary': summary,
            'results': all_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to: {output_path}")
    
    return summary, all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate OriTransfer Compressor')
    
    parser.add_argument('--compressor_checkpoint', type=str, required=True)
    parser.add_argument('--src_compressor_path', type=str, required=True)
    parser.add_argument('--src_decoder_model_path', type=str, required=True)
    parser.add_argument('--tgt_model_path', type=str, required=True)
    parser.add_argument('--n_mem_tokens', type=int, required=True)
    parser.add_argument('--train_mode', type=str, required=True)
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_generate', action='store_true')
    parser.add_argument('--generate_samples', type=int, default=0,
                       help='If >0, only generate for first N samples (still evaluates loss/accuracy for all).')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation (default: 1)')
    
    args = parser.parse_args()
    
    evaluate_model(
        compressor_checkpoint=args.compressor_checkpoint,
        src_compressor_path=args.src_compressor_path,
        src_decoder_model_path=args.src_decoder_model_path,
        tgt_model_path=args.tgt_model_path,
        n_mem_tokens=args.n_mem_tokens,
        train_mode=args.train_mode,
        test_data_path=args.test_data_path,
        output_path=args.output_path,
        max_samples=args.max_samples,
        max_length=args.max_length,
        device=args.device,
        generate_text=not args.no_generate,
        generate_samples=args.generate_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
