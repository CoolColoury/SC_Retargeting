'''
Base trainer for simple compressor
'''

import torch
import copy
from transformers import Trainer


class CompressorTrainer(Trainer):
    """
    Custom trainer for compressor model
    Handles the two-model architecture (compressor + decoder)
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss for compressor training
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
        Temporarily separate shared weights and use safe_serialization=False
        """
        model = self.model
        
        # Store original references for restoration
        compressor_lm_head_orig = None
        decoder_lm_head_orig = None
        
        # Handle compressor shared weights
        if hasattr(model.compressor, 'lm_head') and hasattr(model.compressor, 'model'):
            if hasattr(model.compressor.model, 'embed_tokens'):
                compressor_lm_head = model.compressor.lm_head.weight
                compressor_embed_tokens = model.compressor.model.embed_tokens.weight
                if compressor_lm_head.data_ptr() == compressor_embed_tokens.data_ptr():
                    compressor_lm_head_orig = compressor_lm_head
                    model.compressor.lm_head.weight = torch.nn.Parameter(compressor_lm_head.clone())
        
        # Handle decoder shared weights  
        if hasattr(model.decoder, 'lm_head') and hasattr(model.decoder, 'model'):
            if hasattr(model.decoder.model, 'embed_tokens'):
                decoder_lm_head = model.decoder.lm_head.weight
                decoder_embed_tokens = model.decoder.model.embed_tokens.weight
                if decoder_lm_head.data_ptr() == decoder_embed_tokens.data_ptr():
                    decoder_lm_head_orig = decoder_lm_head
                    model.decoder.lm_head.weight = torch.nn.Parameter(decoder_lm_head.clone())
        
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
            model.compressor.lm_head.weight = compressor_lm_head_orig
        
        if decoder_lm_head_orig is not None:
            model.decoder.lm_head.weight = decoder_lm_head_orig

