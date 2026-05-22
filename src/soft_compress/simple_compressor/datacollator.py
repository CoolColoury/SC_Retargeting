'''
Data collator for simple compressor training
'''

import torch
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CompressorDataCollator:
    """
    Data collator for compressor training
    Handles padding for both compressor input and decoder input
    """
    compressor_tokenizer: any
    decoder_tokenizer: any
    max_length: int = 512
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples
        Each sample should have 'text' field
        """
        texts = [f['text'] for f in features]
        
        # Tokenize for compressor (input to compress)
        compress_inputs = self.compressor_tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        
        # Tokenize for decoder (target to reconstruct)
        decoder_inputs = self.decoder_tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        
        return {
            'compress_input_ids': compress_inputs['input_ids'],
            'compress_attention_mask': compress_inputs['attention_mask'],
            'decoder_input_ids': decoder_inputs['input_ids'],
            'decoder_attention_mask': decoder_inputs['attention_mask']
        }

