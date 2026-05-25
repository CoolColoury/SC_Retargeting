"""
Evaluate transferred compression on target model.
Similar to transfer_exp/evaluate_transfer.py but adapted for compressor transfer results.
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Add paths
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from soft_compress.evaluation.text_generation_scores import cal_bleu_rouge

sys.path.append(str(Path(__file__).parent.parent / 'simple_mem'))

from mem_cell import MemoryCell
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_results(results_path: str) -> tuple:
    """Load transferred compression results"""
    results_file = Path(results_path)
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    if results_file.stat().st_size == 0:
        raise ValueError(f"Results file is empty: {results_path}")
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON from {results_path}\n"
            f"JSON error: {e}"
        ) from e
    
    if not isinstance(data, dict):
        raise ValueError(f"Invalid data format: expected dict, got {type(data)}")
    
    metadata = data.get('metadata', {})
    results = data.get('results', [])

    if not results:
        raise ValueError(f"No results found in {results_path}")

    npz_name = metadata.get('memories_npz')
    if npz_name and results and 'memory' not in results[0]:
        npz_path = results_file.parent / npz_name
        if not npz_path.exists():
            npz_path = results_file.with_suffix('.memories.npz')
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Memory sidecar not found for {results_path}: expected {npz_name}"
            )
        loaded = np.load(npz_path, allow_pickle=True)
        memories = loaded['memories']
        if len(memories) != len(results):
            raise ValueError(
                f"Memory sidecar length mismatch: {len(memories)} vs {len(results)} results"
            )
        for idx, row in enumerate(results):
            row['memory'] = memories[idx]
    
    # For transferred results, model name is in target_model
    model_name = metadata.get('target_model')
    if not model_name:
        raise ValueError(f"No target_model found in metadata")
    
    return results, metadata, model_name


def load_text_from_dataset(text_id: str, dataset_path: str = None) -> str:
    """Load text from original dataset using text_id"""
    if dataset_path is None:
        common_paths = [
            'datasets/simple_mem/fineweb_train.json',
            'datasets/simple_mem/fineweb_all.json',
            'datasets/simple_mem/fineweb_test.json',
            'datasets/simple_mem/texts.json',
        ]
        existing_paths = []
        for path in common_paths:
            p = Path(path)
            if p.exists():
                existing_paths.append((p.stat().st_size, path))
        
        if existing_paths:
            existing_paths.sort(reverse=True)
            dataset_path = existing_paths[0][1]
        else:
            raise ValueError("Could not find dataset file")
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"Dataset file not found: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Try exact match first
    for item in dataset:
        if item.get('id') == text_id:
            return item.get('text', '')
    
    # Try to extract index from text_id
    import re
    match = re.search(r'(\d+)$', text_id)
    if match:
        idx = int(match.group(1))
        if 0 <= idx < len(dataset):
            item = dataset[idx]
            return item.get('text', '')
    
    # Try direct index
    try:
        idx = int(text_id)
        if 0 <= idx < len(dataset):
            return dataset[idx].get('text', '')
    except ValueError:
        pass
    
    raise ValueError(f"Text with id '{text_id}' not found in dataset '{dataset_path}'")


def build_dataset_index(dataset_path: str):
    """Load dataset once and build fast lookup index."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError(f"Invalid dataset format: expected list, got {type(dataset)}")

    id_to_text = {}
    for idx, item in enumerate(dataset):
        if isinstance(item, dict):
            item_id = item.get('id')
            item_text = item.get('text', '')
            if item_id is not None:
                id_to_text[str(item_id)] = item_text
            id_to_text[str(idx)] = item_text
        else:
            id_to_text[str(idx)] = str(item)
    return dataset, id_to_text


def lookup_text_from_index(text_id: str, dataset: list, id_to_text: dict) -> str:
    """Lookup text by id/index from prebuilt dataset index."""
    if text_id in id_to_text:
        return id_to_text[text_id]

    import re
    match = re.search(r'(\d+)$', text_id)
    if match:
        idx = int(match.group(1))
        if 0 <= idx < len(dataset):
            item = dataset[idx]
            if isinstance(item, dict):
                return item.get('text', '')
            return str(item)

    raise ValueError(f"Text with id '{text_id}' not found in preloaded dataset")


def evaluate_reconstruction(
    memory_vector: np.ndarray,
    target_ids: list,
    model_with_memory: MemoryCell,
    tokenizer,
    device: str = 'cuda',
    generate_text: bool = True,
) -> dict:
    """Evaluate reconstruction accuracy and generate text"""
    n_mem_tokens = memory_vector.shape[0]
    
    # Update memory data
    model_with_memory.memory.data = torch.tensor(
        memory_vector, dtype=torch.bfloat16
    ).to(device)
    
    input_ids = torch.tensor([target_ids], dtype=torch.long).to(device)
    
    with torch.inference_mode():
        output, _ = model_with_memory(input_ids=input_ids)
        loss = output.loss.item()
        
        # Token accuracy - move to CPU immediately
        shift_logits = output.logits[:, :-1, :].cpu()
        shift_labels = input_ids[:, 1:].cpu().clone()
        del output
        del input_ids
        torch.cuda.empty_cache()
        
        # Process on CPU
        predictions = torch.argmax(shift_logits, dim=-1)
        correct = (predictions == shift_labels).float()
        accuracy = correct.mean().item()
        
        # Generate reconstructed text
        generated_text = ""
        if generate_text:
            generated_ids = predictions[0].tolist()
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        del shift_logits, shift_labels, predictions, correct
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'num_tokens': len(target_ids),
        'generated_text': generated_text
    }


def evaluate_transfer(
    results_path: str,
    dataset_path: str = None,
    device: str = 'cuda',
    output_dir: str = None,
    eval_batch_size: int = 8,
    generate_text: bool = True,
    generate_samples: int = 0,
) -> dict:
    """
    Evaluate transferred compression on target model.
    
    Args:
        results_path: Path to transferred results JSON (from transfer_compressor)
        dataset_path: Path to original dataset
        device: Device to run on
        output_dir: Base output directory
        eval_batch_size: Batch size for evaluation loop
        generate_text: Whether to generate reconstructed text for BLEU/ROUGE
        generate_samples: If >0, only generate for first N samples
    """
    results, metadata, model_name = load_results(results_path)
    
    print(f"Evaluating {len(results)} transferred vectors on target model")
    print(f"Target model: {model_name}")
    print(f"Source: {metadata.get('source_experiment', 'unknown')}")
    print(f"Transfer method: {metadata.get('transfer_method', 'unknown')}")
    
    # Use dataset from metadata if not provided
    if dataset_path is None:
        dataset_path = metadata.get('dataset_path')
    
    if dataset_path:
        print(f"Dataset: {dataset_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    # print("model_name:", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True
    )
    model.eval()
    torch.cuda.empty_cache()
    
    # Determine memory shape
    if not results:
        raise ValueError("No results to evaluate")
    first_memory = np.array(results[0]['memory'])
    n_mem_tokens = first_memory.shape[0]
    memory_dim = first_memory.shape[1]
    
    print(f"Creating MemoryCell (n_mem_tokens={n_mem_tokens}, memory_dim={memory_dim})...")
    model_with_memory = MemoryCell(model, n_mem_tokens, memory_dim)
    model_with_memory.eval()
    
    eval_results = []
    bleu_list, bleu1_list, bleu2_list = [], [], []
    bleu3_list, bleu4_list, rougeL_list = [], [], []
    generated_count = 0

    eval_batch_size = max(1, int(eval_batch_size))
    generate_samples = max(0, int(generate_samples))
    print(f"Eval batch size: {eval_batch_size}")
    print(f"Generate text: {bool(generate_text)}")
    if generate_text and generate_samples > 0:
        print(f"Generate samples limit: {generate_samples}")

    if dataset_path is None:
        raise ValueError("dataset_path is required for evaluation")

    print("Loading dataset index once...")
    dataset_data, id_to_text = build_dataset_index(dataset_path)

    tqdm_prepare = tqdm(
        results,
        desc="Preparing eval samples",
        mininterval=1.0,
        dynamic_ncols=True,
    )
    prepared_samples = []
    for idx, result in enumerate(tqdm_prepare):
        text_id = result['text_id']
        memory = np.array(result['memory'])

        try:
            text = lookup_text_from_index(text_id, dataset_data, id_to_text)
            max_length = metadata.get('max_length', 512)
            tokens = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
            target_ids = tokens['input_ids'][0].tolist()
            truncated_text = tokenizer.decode(target_ids, skip_special_tokens=True)
        except Exception as e:
            if idx < 3:
                print(f"  Warning: Could not process {text_id}: {e}")
            continue

        prepared_samples.append({
            'text_id': text_id,
            'memory': memory,
            'text': text,
            'target_ids': target_ids,
            'truncated_text': truncated_text,
        })
    
    try:
        tqdm_eval = tqdm(
            range(0, len(prepared_samples), eval_batch_size),
            desc="Evaluating batches",
            mininterval=1.0,
            dynamic_ncols=True,
        )
        for batch_idx, batch_start in enumerate(tqdm_eval, start=1):
            batch_samples = prepared_samples[batch_start: batch_start + eval_batch_size]
            for sample in batch_samples:
                text_id = sample['text_id']
                memory = sample['memory']
                text = sample['text']
                target_ids = sample['target_ids']
                truncated_text = sample['truncated_text']

                # Verify memory shape
                if memory.shape != (n_mem_tokens, memory_dim):
                    print(f"  Warning: Memory shape mismatch for {text_id}")
                    n_mem_tokens, memory_dim = memory.shape[0], memory.shape[1]
                    model_with_memory = MemoryCell(model, n_mem_tokens, memory_dim)
                    model_with_memory.eval()

                should_generate = bool(generate_text) and (generate_samples == 0 or generated_count < generate_samples)
                metrics = evaluate_reconstruction(
                    memory_vector=memory,
                    target_ids=target_ids,
                    model_with_memory=model_with_memory,
                    tokenizer=tokenizer,
                    device=device,
                    generate_text=should_generate,
                )

                # Calculate BLEU/ROUGE only when text generation is enabled for this sample
                generated_text = metrics['generated_text']
                if should_generate:
                    text_metrics = cal_bleu_rouge(generated_text, truncated_text)
                    bleu_list.append(text_metrics['bleu'])
                    bleu1_list.append(text_metrics['bleu1'])
                    bleu2_list.append(text_metrics['bleu2'])
                    bleu3_list.append(text_metrics['bleu3'])
                    bleu4_list.append(text_metrics['bleu4'])
                    rougeL_list.append(text_metrics['rougeL'])
                    generated_count += 1
                else:
                    text_metrics = {
                        'bleu': 0.0, 'bleu1': 0.0, 'bleu2': 0.0,
                        'bleu3': 0.0, 'bleu4': 0.0, 'rougeL': 0.0
                    }

                eval_results.append({
                    'text_id': text_id,
                    'original_text': text,
                    'truncated_text': truncated_text,
                    'generated_text': generated_text,
                    'transfer_accuracy': metrics['accuracy'],
                    'transfer_loss': metrics['loss'],
                    'num_tokens': metrics['num_tokens'],
                    'text_generated': bool(should_generate),
                    **text_metrics
                })

                del metrics, generated_text, text_metrics, memory

            # Keep progress visible in redirected logs as well.
            if batch_idx % 10 == 0:
                print(f"Evaluated {min(batch_start + len(batch_samples), len(prepared_samples))}/{len(prepared_samples)} samples")

            torch.cuda.empty_cache()
    
    finally:
        print("Cleaning up model...")
        del model_with_memory, model
        torch.cuda.empty_cache()
    
    # Summary
    successful = [r for r in eval_results if 'transfer_accuracy' in r]
    avg_transfer_acc = np.mean([r['transfer_accuracy'] for r in successful])
    
    summary = {
        'source_experiment': metadata.get('source_experiment'),
        'source_model': metadata.get('source_model'),
        'target_model': model_name,
        'transfer_method': metadata.get('transfer_method'),
        'dataset_path': metadata.get('dataset_path'),
        'num_texts': len(eval_results),
        'generate_text': bool(generate_text),
        'generate_samples': int(generate_samples),
        'num_generated_texts': int(generated_count),
        'avg_transfer_accuracy': float(avg_transfer_acc),
        'avg_bleu': float(np.mean(bleu_list)) if bleu_list else 0.0,
        'avg_bleu1': float(np.mean(bleu1_list)) if bleu1_list else 0.0,
        'avg_bleu2': float(np.mean(bleu2_list)) if bleu2_list else 0.0,
        'avg_bleu3': float(np.mean(bleu3_list)) if bleu3_list else 0.0,
        'avg_bleu4': float(np.mean(bleu4_list)) if bleu4_list else 0.0,
        'avg_rougeL': float(np.mean(rougeL_list)) if rougeL_list else 0.0,
        'results': eval_results,
    }
    
    # Save results
    if output_dir is None:
        output_dir = Path('outputs/compressor_transfer_eval')
    else:
        output_dir = Path(output_dir)
    
    results_path = Path(results_path)
    output_path = output_dir / results_path.parent.name
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"eval_transfer_{results_path.stem}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation saved to: {output_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate transferred compression on target model')
    parser.add_argument('--results_path', type=str, required=True,
                       help='Path to transferred results JSON')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='Path to original dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/compressor_transfer_eval',
                       help='Base output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    parser.add_argument('--eval_batch_size', type=int, default=8,
                       help='Batch size for evaluation loop')
    parser.add_argument('--generate_text', type=int, default=1, choices=[0, 1],
                       help='Whether to generate reconstructed text (1=yes, 0=no)')
    parser.add_argument('--generate_samples', type=int, default=0,
                       help='If >0, only generate text for first N samples (0=all)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Evaluate Transfer (Compressor)")
    print("=" * 60)
    
    summary = evaluate_transfer(
        results_path=args.results_path,
        dataset_path=args.dataset_path,
        device=args.device,
        output_dir=args.output_dir,
        eval_batch_size=args.eval_batch_size,
        generate_text=bool(args.generate_text),
        generate_samples=args.generate_samples,
    )
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Source: {summary['source_model']}")
    print(f"Target: {summary['target_model']}")
    print(f"Transfer method: {summary['transfer_method']}")
    print(f"Avg transfer accuracy: {summary['avg_transfer_accuracy']:.4f}")
    print(f"Avg BLEU: {summary['avg_bleu']:.4f}")
    print(f"Avg BLEU-4: {summary['avg_bleu4']:.4f}")
    print(f"Avg ROUGE-L: {summary['avg_rougeL']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
