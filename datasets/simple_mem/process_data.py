"""
Cursor generated code and modified by author.
Process FineWeb dataset to create training and test datasets.

This script:
1. Loads the FineWeb dataset from local file or huggingface(HuggingFaceFW/fineweb)
2. Filters texts with at least 1000 tokens
3. Selects 200 texts
4. Splits into 100 training and 100 test samples
5. Saves as JSON files
"""

import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import random


def count_tokens(text: str, tokenizer) -> int:
    """Count the number of tokens in a text."""
    tokens = tokenizer(text, return_tensors='pt', add_special_tokens=False)
    return tokens['input_ids'].shape[1]


def filter_texts_by_token_count(texts: list, tokenizer, min_tokens: int = 1000) -> list:
    """
    Filter texts that have at least min_tokens tokens.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer to use for counting tokens
        min_tokens: Minimum number of tokens required
        
    Returns:
        List of filtered texts that meet the token requirement
    """
    filtered_texts = []
    
    print(f"Filtering texts with at least {min_tokens} tokens...")
    for text in tqdm(texts, desc="Filtering texts"):
        if text and isinstance(text, str) and len(text.strip()) > 0:
            try:
                num_tokens = count_tokens(text, tokenizer)
                if num_tokens >= min_tokens:
                    filtered_texts.append({
                        'text': text,
                        'token_count': num_tokens
                    })
            except Exception as e:
                # Skip texts that cause tokenization errors
                continue
    
    return filtered_texts


def process_fineweb_dataset(
    output_dir: str = "../simple_mem_500",
    min_tokens: int = 300,
    total_samples: int = 1000,
    train_samples: int = 500,
    test_samples: int = 500,
    tokenizer_name: str = "${MODELS_DIR}/Llama-3-8B-Instruct",
    seed: int = 42,
    max_check_sample: int = 2050000
):
    """
    Process FineWeb dataset to create training and test datasets.
    
    Args:
        output_dir: Directory to save output files
        min_tokens: Minimum number of tokens per text
        total_samples: Total number of samples to select
        train_samples: Number of training samples
        test_samples: Number of test samples
        tokenizer_name: Tokenizer to use for counting tokens
        seed: Random seed for reproducibility
        max_check_sample: Maximum number of samples to check from dataset
    """
    # Set random seed
    random.seed(seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("FineWeb Dataset Processing")
    print("="*80)
    print(f"Output directory: {output_path}")
    print(f"Minimum tokens per text: {min_tokens}")
    print(f"Total samples: {total_samples}")
    print(f"Training samples: {train_samples}")
    print(f"Test samples: {test_samples}")
    print(f"Tokenizer: {tokenizer_name}")
    print("="*80)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer {tokenizer_name}: {e}")
        print("Falling back to GPT-2 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Load FineWeb dataset from local file
    try:
        # Try to load the dataset
        # Note: This may take some time depending on your internet connection
        dataset = load_dataset("${PROJECT_ROOT}/datasets/fineweb-data-10BT", name="default")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    

    # Only work with the 'train' split, handle DatasetDict properly
    if isinstance(dataset, dict) or hasattr(dataset, "keys"):
        # Assume DatasetDict, use 'train' split
        dataset_split = dataset["train"]
    else:
        dataset_split = dataset
    dataset_split = dataset_split.select_columns(["text"])
    dataset_split = dataset_split.select(range(max_check_sample))
    dataset = dataset_split

    # Filter texts by token count
    print(f"\nFiltering texts with at least {min_tokens} tokens...")
    filtered_dataset = dataset.filter(lambda x: count_tokens(x['text'], tokenizer) >= min_tokens, num_proc=32)

    print(f"\nFound {len(filtered_dataset)} texts meeting the token requirement")
    
    if len(filtered_dataset) < total_samples:
        print(f"Warning: Only found {len(filtered_dataset)} texts with >= {min_tokens} tokens")
        print(f"Requested {total_samples} samples, will use all available texts")
        total_samples = len(filtered_dataset)
        # Adjust train/test split proportionally
        train_samples = int(total_samples * train_samples / (train_samples + test_samples))
        test_samples = total_samples - train_samples
    
    # Randomly sample texts
    print(f"\nRandomly sampling {total_samples} texts...")
    filtered_texts = filtered_dataset.to_list()
    sampled_texts = random.sample(filtered_texts, total_samples)
    
    # Split into train and test
    random.shuffle(sampled_texts)
    train_texts = sampled_texts[:train_samples]
    test_texts = sampled_texts[train_samples:train_samples + test_samples]
    
    print(f"\nSplit into:")
    print(f"  Training set: {len(train_texts)} texts")
    print(f"  Test set: {len(test_texts)} texts")
    
    # Format as JSON (matching the existing format)
    def format_as_json(texts_list, prefix="text"):
        return [
            {
                "id": f"{prefix}_{i}",
                "text": item['text']
            }
            for i, item in enumerate(texts_list)
        ]
    
    train_json = format_as_json(train_texts, prefix="train")
    test_json = format_as_json(test_texts, prefix="test")
    all_json = format_as_json(sampled_texts, prefix="fineweb")
    
    # Save to files
    print(f"\nSaving datasets to {output_path}...")
    
    train_file = output_path / "fineweb_train.json"
    test_file = output_path / "fineweb_test.json"
    all_file = output_path / "fineweb_all.json"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_json, f, indent=2, ensure_ascii=False)
    print(f"  Saved training set: {train_file} ({len(train_json)} texts)")
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_json, f, indent=2, ensure_ascii=False)
    print(f"  Saved test set: {test_file} ({len(test_json)} texts)")
    
    with open(all_file, 'w', encoding='utf-8') as f:
        json.dump(all_json, f, indent=2, ensure_ascii=False)
    print(f"  Saved all samples: {all_file} ({len(all_json)} texts)")
    
    # Save statistics
    stats = {
        "total_samples": total_samples,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "min_tokens": min_tokens,
        "tokenizer": tokenizer_name,
    }
    
    stats_file = output_path / "fineweb_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Saved statistics: {stats_file}")
    
    print("\n" + "="*80)
    print("Processing complete!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - {train_file.name}: Training set ({train_samples} texts)")
    print(f"  - {test_file.name}: Test set ({test_samples} texts)")
    print(f"  - {all_file.name}: All samples ({total_samples} texts)")
    print(f"  - {stats_file.name}: Statistics")
    print("\nYou can now use these files for your compression experiments.")


def main():
    process_fineweb_dataset()


if __name__ == '__main__':
    main()

