"""
OriTransfer Compressor Model: Direct Simple Compressor Continuation Training

Different from TransferCompressor (Simple Compressor + Projector):
- Reuses the entire SimpleCompressor structure
- Directly continues training from source compressor to target decoder
- Example: gpt2-to-llama1b -> gpt2-to-llama8b
- Two training modes:
  1. Train only Converter
  2. Train Encoder + Converter
- If dimension mismatch, creates a new Converter
"""

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from pathlib import Path
import sys

# Add simple_compressor to path
sys.path.append(str(Path(__file__).parent.parent / 'simple_compressor'))
from simple_compressor import SimpleCompressor, Projector, LlamaRMSNorm
from decoder_prefix_tokens import decoder_bos_token_id


class OriTransferCompressor(nn.Module):
    """
    OriTransfer Compressor: Direct continuation training of Simple Compressor
    
    Architecture:
    - Loads source SimpleCompressor (e.g., gpt2-to-llama1b)
    - Replaces target decoder with new target (e.g., llama8b)
    - Optionally recreates Converter if dimension mismatch
    - Supports two training modes:
      1. Train only Converter (freeze Encoder)
      2. Train Encoder + Converter
    """
    def __init__(
        self,
        src_compressor_path: str,
        src_decoder_model_path: str,
        tgt_model_path: str,
        n_mem_tokens: int,
        train_mode: str = 'converter_only',  # 'converter_only' or 'encoder_converter'
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        Args:
            src_compressor_path: Path to trained source compressor checkpoint
            src_decoder_model_path: Source decoder model path (for loading source compressor)
            tgt_model_path: Target model path (new decoder)
            n_mem_tokens: Number of memory tokens
            train_mode: Training mode - 'converter_only' or 'encoder_converter'
            dtype: Data type for models
        """
        super().__init__()
        
        print("=" * 60)
        print("Initializing OriTransfer Compressor")
        print("=" * 60)
        print(f"Training mode: {train_mode}")
        
        # Validate training mode
        if train_mode not in ['converter_only', 'encoder_converter']:
            raise ValueError(f"Invalid train_mode: {train_mode}. Must be 'converter_only' or 'encoder_converter'")
        
        self.train_mode = train_mode
        self.n_mem_tokens = n_mem_tokens
        
        # Step 1: Load source SimpleCompressor
        print(f"[1/4] Loading source compressor from: {src_compressor_path}")
        src_compressor = SimpleCompressor.from_pretrained(
            checkpoint_path=src_compressor_path,
            decoder_model_name=src_decoder_model_path,
            n_mem_tokens=n_mem_tokens,
            use_lora=False,
            dtype=dtype,
            device='cpu'
        )
        
        # Extract components
        self.compressor = src_compressor.compressor
        self.compressor_tokenizer = src_compressor.compressor_tokenizer
        self.mem_ids = src_compressor.mem_ids
        self.compressor_hidden_size = src_compressor.compressor_hidden_size
        
        # Step 2: Get target decoder config to check dimension
        print(f"[2/4] Loading target decoder config from: {tgt_model_path}")
        tgt_config = AutoConfig.from_pretrained(tgt_model_path)
        tgt_hidden_size = tgt_config.hidden_size
        
        src_hidden_size = src_compressor.decoder_hidden_size
        print(f"       Source decoder hidden_size: {src_hidden_size}")
        print(f"       Target decoder hidden_size: {tgt_hidden_size}")
        
        # Step 3: Create or reuse Converter
        if src_hidden_size != tgt_hidden_size:
            print(f"[3/4] Dimension mismatch! Creating NEW Converter ({self.compressor_hidden_size} -> {tgt_hidden_size})...")
            self.projector = Projector(self.compressor_hidden_size, tgt_hidden_size)
            self.dimension_matched = False
        else:
            print(f"[3/4] Dimension matched. Reusing source Converter...")
            self.projector = src_compressor.projector
            self.dimension_matched = True
        
        # Set projector trainability
        print(f"       Converter is TRAINABLE")
        for param in self.projector.parameters():
            param.requires_grad = True
        
        # Step 4: Set encoder trainability based on mode
        if train_mode == 'converter_only':
            print(f"[3/4] Freezing Encoder (Compressor)...")
            for param in self.compressor.parameters():
                param.requires_grad = False
            self.compressor.eval()
        else:  # encoder_converter
            print(f"[3/4] Encoder (Compressor) is TRAINABLE...")
            for param in self.compressor.parameters():
                param.requires_grad = True
            self.compressor.train()
        
        # Step 5: Load target decoder (frozen)
        print(f"[4/4] Loading target decoder from: {tgt_model_path}")
        print(f"        (This may take a while for large models...)")
        self.decoder = AutoModelForCausalLM.from_pretrained(
            tgt_model_path,
            torch_dtype=dtype
        )
        
        # Freeze decoder
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.eval()
        
        # Load target tokenizer
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(tgt_model_path)
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token
        
        # Store metadata
        self.decoder_hidden_size = tgt_hidden_size
        
        # Cache for faster forward pass
        self._decoder_embeddings = None
        self._decoder_dtype = None
        self._bos_token_id = None
        
        print("=" * 60)
        print("OriTransfer Compressor initialized successfully")
        print("=" * 60)
        
        self.print_parameter_status()
    
    def print_parameter_status(self):
        """Print parameter status for all components"""
        print("\nParameter Status:")
        print("-" * 60)
        
        # Encoder (Compressor)
        enc_trainable = sum(p.numel() for p in self.compressor.parameters() if p.requires_grad)
        enc_total = sum(p.numel() for p in self.compressor.parameters())
        print(f"Encoder (Compressor): {enc_trainable:,} trainable / {enc_total:,} total")
        if self.train_mode == 'converter_only':
            if enc_trainable > 0:
                print("  ⚠️  WARNING: Encoder should be frozen in converter_only mode!")
            else:
                print("  ✓ Encoder is frozen (converter_only mode)")
        else:
            if enc_trainable == 0:
                print("  ⚠️  WARNING: Encoder should be trainable in encoder_converter mode!")
            else:
                print("  ✓ Encoder is trainable (encoder_converter mode)")
        
        # Converter (Projector)
        conv_trainable = sum(p.numel() for p in self.projector.parameters() if p.requires_grad)
        conv_total = sum(p.numel() for p in self.projector.parameters())
        print(f"Converter (Projector): {conv_trainable:,} trainable / {conv_total:,} total")
        if conv_trainable == 0:
            print("  ⚠️  WARNING: Converter has NO trainable parameters!")
        else:
            print("  ✓ Converter is trainable")
        if not self.dimension_matched:
            print("  ℹ️  Note: New Converter created due to dimension mismatch")
        
        # Decoder
        dec_trainable = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        dec_total = sum(p.numel() for p in self.decoder.parameters())
        print(f"Target Decoder: {dec_trainable:,} trainable / {dec_total:,} total")
        if dec_trainable > 0:
            print("  ⚠️  WARNING: Decoder has trainable parameters!")
        else:
            print("  ✓ Decoder is frozen")
        
        print("-" * 60)
        
        # Total trainable
        total_trainable = enc_trainable + conv_trainable + dec_trainable
        total = enc_total + conv_total + dec_total
        print(f"TOTAL: {total_trainable:,} trainable / {total:,} total ({100*total_trainable/total:.2f}%)")
        print("-" * 60)
    
    def compress(self, input_ids, attention_mask):
        """
        Compress input using compressor model
        Returns: compressed memory tokens (batch_size, n_mem_tokens, decoder_hidden_size)
        """
        # Append memory tokens to input
        batch_size = input_ids.size(0)
        mem_ids_tensor = torch.tensor(self.mem_ids, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
        input_ids_with_mem = torch.cat((input_ids, mem_ids_tensor), dim=1)
        
        # Create attention mask for the extended sequence
        mem_attention = torch.ones((batch_size, self.n_mem_tokens), dtype=attention_mask.dtype, device=attention_mask.device)
        full_attention_mask = torch.cat((attention_mask, mem_attention), dim=1)
        
        # Forward through compressor
        if self.train_mode == 'converter_only':
            # Encoder frozen, use no_grad for efficiency
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.compressor(
                        input_ids=input_ids_with_mem,
                        attention_mask=full_attention_mask,
                        output_hidden_states=True,
                        return_dict=True
                    )
        else:
            # Encoder trainable
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.compressor(
                    input_ids=input_ids_with_mem,
                    attention_mask=full_attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
        
        # Extract hidden states at memory token positions
        hidden_states = outputs.hidden_states[-1]
        compressed = hidden_states[:, -self.n_mem_tokens:, :]
        
        # Project through converter (always trainable)
        compressed = self.projector(compressed)
        
        return compressed
    
    def forward(self, compress_input_ids, compress_attention_mask, 
                decoder_input_ids, decoder_attention_mask):
        """
        Forward pass for training
        
        Args:
            compress_input_ids: Input to compress (batch_size, seq_len)
            compress_attention_mask: Attention mask for compression (batch_size, seq_len)
            decoder_input_ids: Target text for reconstruction (batch_size, target_len)
            decoder_attention_mask: Attention mask for target (batch_size, target_len)
            
        Returns:
            CausalLMOutputWithPast with loss and logits
        """
        # Step 1: Compress input
        compressed_memory = self.compress(compress_input_ids, compress_attention_mask)
        
        # Step 2: Get decoder embeddings (cached for performance)
        if self._decoder_embeddings is None:
            self._decoder_embeddings = self.decoder.get_input_embeddings()
            self._decoder_dtype = self._decoder_embeddings.weight.dtype
            self._bos_token_id = decoder_bos_token_id(
                self.decoder_tokenizer, self.decoder.config
            )
        
        decoder_embeddings = self._decoder_embeddings
        target_dtype = self._decoder_dtype
        
        # Ensure compressed memory matches decoder dtype
        compressed_memory = compressed_memory.to(dtype=target_dtype)
        
        # Step 3: Get BOS embedding
        bos_embedding = decoder_embeddings(
            torch.tensor([[self._bos_token_id]], device=compress_input_ids.device)
        ).expand(compress_input_ids.shape[0], -1, -1)
        
        # Step 4: Get target embeddings
        target_embeddings = decoder_embeddings(decoder_input_ids)
        
        # Step 5: Concatenate: [BOS] + [Memory] + [Target]
        inputs_embeds = torch.cat([bos_embedding, compressed_memory, target_embeddings], dim=1)
        
        # Step 6: Prepare attention mask
        batch_size = compress_input_ids.shape[0]
        bos_mask = torch.ones(batch_size, 1, dtype=decoder_attention_mask.dtype, device=compress_input_ids.device)
        memory_mask = torch.ones(batch_size, self.n_mem_tokens, dtype=decoder_attention_mask.dtype, device=compress_input_ids.device)
        full_attention_mask = torch.cat([bos_mask, memory_mask, decoder_attention_mask], dim=1)
        
        # Step 7: Prepare labels: [-100] * (1 + n_mem) + [Target]
        prefix_labels = torch.full(
            (batch_size, self.n_mem_tokens + 1),
            -100,
            dtype=decoder_input_ids.dtype,
            device=compress_input_ids.device
        )
        target_labels = decoder_input_ids.clone()
        target_labels[decoder_attention_mask == 0] = -100
        labels = torch.cat([prefix_labels, target_labels], dim=1)
        
        # Step 8: Ensure proper training modes
        if self.train_mode == 'converter_only':
            if self.compressor.training:
                self.compressor.eval()
        else:
            if not self.compressor.training:
                self.compressor.train()
        
        if self.decoder.training:
            self.decoder.eval()
        
        # Step 9: Forward through frozen decoder
        with torch.amp.autocast("cuda", dtype=target_dtype):
            outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                labels=labels,
                return_dict=True
            )
        
        return CausalLMOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits
        )
    
    @classmethod
    def from_pretrained(cls, checkpoint_path, **kwargs):
        """Load from checkpoint"""
        checkpoint_path_obj = Path(checkpoint_path)
        
        # Try to load config
        config_path = checkpoint_path_obj / 'ori_transfer_config.json'
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Override with kwargs if provided (kwargs take precedence)
            # This is important because config may have 'saved_with_checkpoint' placeholders
            for key in ['src_compressor_path', 'src_decoder_model_path', 'tgt_model_path']:
                if key in kwargs and kwargs[key] != 'saved_with_checkpoint':
                    config[key] = kwargs[key]
            
            # Use kwargs for other parameters too
            if 'n_mem_tokens' in kwargs:
                config['n_mem_tokens'] = kwargs['n_mem_tokens']
            if 'train_mode' in kwargs:
                config['train_mode'] = kwargs['train_mode']
            
            # Check if we have valid paths (not placeholders)
            if config.get('src_compressor_path') == 'saved_with_checkpoint':
                if 'src_compressor_path' not in kwargs:
                    raise ValueError(
                        f"Config file has placeholder 'saved_with_checkpoint' for src_compressor_path. "
                        f"Please provide src_compressor_path in kwargs."
                    )
                config['src_compressor_path'] = kwargs['src_compressor_path']
            
            if config.get('src_decoder_model_path') == 'saved_with_checkpoint':
                if 'src_decoder_model_path' not in kwargs:
                    raise ValueError(
                        f"Config file has placeholder 'saved_with_checkpoint' for src_decoder_model_path. "
                        f"Please provide src_decoder_model_path in kwargs."
                    )
                config['src_decoder_model_path'] = kwargs['src_decoder_model_path']
            
            if config.get('tgt_model_path') == 'saved_with_checkpoint':
                if 'tgt_model_path' not in kwargs:
                    raise ValueError(
                        f"Config file has placeholder 'saved_with_checkpoint' for tgt_model_path. "
                        f"Please provide tgt_model_path in kwargs."
                    )
                config['tgt_model_path'] = kwargs['tgt_model_path']
            
            model = cls(
                src_compressor_path=config['src_compressor_path'],
                src_decoder_model_path=config['src_decoder_model_path'],
                tgt_model_path=config['tgt_model_path'],
                n_mem_tokens=config['n_mem_tokens'],
                train_mode=config.get('train_mode', 'converter_only'),
                dtype=kwargs.get('dtype', torch.bfloat16)
            )
            
            # Load state dict
            checkpoint_file = checkpoint_path_obj / 'pytorch_model.bin'
            if checkpoint_file.exists():
                saved_state_dict = torch.load(checkpoint_file, map_location='cpu')
                
                # Debug: Print what's being loaded
                print(f"  Loading state dict with {len(saved_state_dict)} keys")
                
                # Get model state dict to check what we're loading into
                model_state = model.state_dict()
                
                # Filter out keys that don't exist in model (for debugging)
                keys_to_load = {}
                keys_not_in_model = []
                for key, value in saved_state_dict.items():
                    if key in model_state:
                        # Check shape match
                        if model_state[key].shape == value.shape:
                            keys_to_load[key] = value
                        else:
                            print(f"  ⚠ Shape mismatch for {key}: saved {value.shape} vs model {model_state[key].shape}")
                            keys_not_in_model.append(key)
                    else:
                        keys_not_in_model.append(key)
                
                if keys_not_in_model:
                    print(f"  ⚠ Warning: {len(keys_not_in_model)} keys from checkpoint not found in model:")
                    for key in keys_not_in_model[:5]:
                        print(f"    - {key}")
                    if len(keys_not_in_model) > 5:
                        print(f"    ... and {len(keys_not_in_model) - 5} more")
                
                # Load the filtered state dict
                if keys_to_load:
                    # Before loading, check if projector weights exist and match
                    projector_keys = [k for k in keys_to_load.keys() if k.startswith('projector.')]
                    if projector_keys:
                        print(f"  Found {len(projector_keys)} projector keys to load")
                        # Check first projector weight to verify shape
                        first_proj_key = projector_keys[0]
                        if first_proj_key in model_state:
                            print(f"    Checking {first_proj_key}:")
                            print(f"      Saved shape: {keys_to_load[first_proj_key].shape}")
                            print(f"      Model shape: {model_state[first_proj_key].shape}")
                    
                    missing_keys, unexpected_keys = model.load_state_dict(keys_to_load, strict=False)
                    
                    if missing_keys:
                        print(f"  ⚠ Warning: {len(missing_keys)} model keys not found in checkpoint:")
                        for key in missing_keys[:5]:
                            print(f"    - {key}")
                        if len(missing_keys) > 5:
                            print(f"    ... and {len(missing_keys) - 5} more")
                    
                    if unexpected_keys:
                        print(f"  ⚠ Warning: {len(unexpected_keys)} unexpected keys (should not happen):")
                        for key in unexpected_keys[:5]:
                            print(f"    - {key}")
                        if len(unexpected_keys) > 5:
                            print(f"    ... and {len(unexpected_keys) - 5} more")
                    
                    print(f"  ✓ Successfully loaded {len(keys_to_load)}/{len(saved_state_dict)} keys")
                    
                    # Verify critical components were loaded
                    compressor_loaded = any(k.startswith('compressor.') for k in keys_to_load.keys())
                    projector_loaded = any(k.startswith('projector.') for k in keys_to_load.keys())
                    
                    print(f"  ✓ Compressor weights loaded: {compressor_loaded}")
                    print(f"  ✓ Projector weights loaded: {projector_loaded}")
                    
                    # Verify weights were actually updated (check a sample weight)
                    if projector_loaded and projector_keys:
                        sample_key = projector_keys[0]
                        # Compare loaded weight with model weight (cast to same dtype for safety)
                        loaded_weight = keys_to_load[sample_key].to(torch.float32)
                        model_weight_after = model.state_dict()[sample_key].to(torch.float32)
                        diff = (loaded_weight - model_weight_after).abs()
                        max_diff = diff.max().item()
                        mean_diff = diff.mean().item()
                        # Check if they're the same (within numerical precision)
                        if torch.allclose(loaded_weight, model_weight_after, atol=1e-5):
                            print(f"  ✓ Verified: Projector weight '{sample_key}' correctly loaded")
                            print(f"    Max difference: {max_diff:.6f}")
                            print(f"    Mean difference: {mean_diff:.6f}")
                        else:
                            print(f"  ✗ Warning: Projector weight '{sample_key}' may not have been loaded correctly!")
                            print(f"    Max difference: {max_diff:.6f}")
                            print(f"    Mean difference: {mean_diff:.6f}")
                else:
                    print(f"  ✗ Error: No keys could be loaded! Check key names match.")
            else:
                print(f"  ⚠ Warning: Checkpoint file not found: {checkpoint_file}")
            
            return model
        else:
            # Config file doesn't exist, try to use kwargs instead
            required_kwargs = ['src_compressor_path', 'src_decoder_model_path', 'tgt_model_path', 'n_mem_tokens']
            missing_kwargs = [kw for kw in required_kwargs if kw not in kwargs]
            
            if missing_kwargs:
                raise FileNotFoundError(
                    f"Config file not found: {config_path}\n"
                    f"Please provide necessary kwargs: {', '.join(required_kwargs)}\n"
                    f"Missing: {', '.join(missing_kwargs)}"
                )
            
            # Use kwargs to create model
            print(f"Config file not found, using provided kwargs to initialize model")
            model = cls(
                src_compressor_path=kwargs['src_compressor_path'],
                src_decoder_model_path=kwargs['src_decoder_model_path'],
                tgt_model_path=kwargs['tgt_model_path'],
                n_mem_tokens=kwargs['n_mem_tokens'],
                train_mode=kwargs.get('train_mode', 'converter_only'),
                dtype=kwargs.get('dtype', torch.bfloat16)
            )
            
            # Try to load state dict if checkpoint exists
            checkpoint_file = checkpoint_path_obj / 'pytorch_model.bin'
            if checkpoint_file.exists():
                saved_state_dict = torch.load(checkpoint_file, map_location='cpu')
                print(f"  Loading state dict with {len(saved_state_dict)} keys")
                
                model_state = model.state_dict()
                keys_to_load = {}
                keys_not_in_model = []
                
                for key, value in saved_state_dict.items():
                    if key in model_state:
                        if model_state[key].shape == value.shape:
                            keys_to_load[key] = value
                        else:
                            print(f"  ⚠ Shape mismatch for {key}: saved {value.shape} vs model {model_state[key].shape}")
                            keys_not_in_model.append(key)
                    else:
                        keys_not_in_model.append(key)
                
                if keys_not_in_model:
                    print(f"  ⚠ Warning: {len(keys_not_in_model)} keys from checkpoint not found in model")
                
                if keys_to_load:
                    missing_keys, unexpected_keys = model.load_state_dict(keys_to_load, strict=False)
                    if missing_keys:
                        print(f"  ⚠ Warning: {len(missing_keys)} model keys not found in checkpoint")
                    if unexpected_keys:
                        print(f"  ⚠ Warning: {len(unexpected_keys)} unexpected keys")
                    print(f"  ✓ Successfully loaded {len(keys_to_load)}/{len(saved_state_dict)} keys")
                else:
                    print(f"  ✗ Error: No keys could be loaded!")
            else:
                print(f"  ⚠ Warning: Checkpoint file not found: {checkpoint_file}")
            
            return model
    
    def save_pretrained(self, save_path):
        """Save model checkpoint"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        import json
        config = {
            'src_compressor_path': 'saved_with_checkpoint',
            'src_decoder_model_path': 'saved_with_checkpoint',
            'tgt_model_path': 'saved_with_checkpoint',
            'n_mem_tokens': self.n_mem_tokens,
            'train_mode': self.train_mode,
            'dimension_matched': self.dimension_matched,
            'compressor_hidden_size': self.compressor_hidden_size,
            'decoder_hidden_size': self.decoder_hidden_size,
        }
        
        with open(save_path / 'ori_transfer_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save trainable weights only
        # Depending on train_mode, save compressor and/or projector
        state_dict = {}
        
        if self.train_mode == 'encoder_converter':
            # Save both encoder and converter
            for name, param in self.compressor.named_parameters():
                if param.requires_grad:
                    state_dict[f'compressor.{name}'] = param.cpu()
        
        # Always save converter
        for name, param in self.projector.named_parameters():
            state_dict[f'projector.{name}'] = param.cpu()
        
        torch.save(state_dict, save_path / 'pytorch_model.bin')
        
        print(f"✓ OriTransfer compressor saved to: {save_path}")
        print(f"  Saved {len(state_dict)} parameters")
