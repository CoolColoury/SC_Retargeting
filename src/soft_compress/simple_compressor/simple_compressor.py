'''
Simple Compressor: Use a PLM to compress input text into memory tokens
'''

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import LoraConfig, TaskType, get_peft_model
from pathlib import Path

# Try to import safetensors, fallback if not available
try:
    from safetensors.torch import load_file as safetensors_load
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_state):
        input_dtype = hidden_state.dtype
        hidden_state = hidden_state.to(torch.float32)
        variance = hidden_state.pow(2).mean(-1, keepdim=True)
        hidden_state = hidden_state * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_state.to(input_dtype)

class Projector(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.dense_in = nn.Linear(input_size, output_size)
        self.dense_out = nn.Linear(output_size, output_size)
        self.RMSNorm = LlamaRMSNorm(input_size)

        self.print_trainable_parameters()
    
    def print_trainable_parameters(self):
        trainable_param = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_param += param.numel()
        print(f"Converter trainable parameters: {trainable_param}, All parameters: {all_param}")
        
    def forward(
        self, 
        embeddings: torch.Tensor
    ):
        embeddings = self.RMSNorm(embeddings)
        # Ensure embeddings and weights have matching dtype
        # Convert to the dtype of dense_in weights
        embeddings = embeddings.to(self.dense_in.weight.dtype)
        x = self.dense_in(embeddings)
        x = self.dense_out(nn.functional.gelu(x))
        # Keep output in the same dtype as dense_out weights
        return x

class SimpleCompressor(nn.Module):
    def __init__(self, compressor_model_name, decoder_model_name, n_mem_tokens, 
                 use_lora=False, lora_config=None, dtype=torch.bfloat16):
        super().__init__()
        
        # Load compressor (trainable PLM)
        print(f"  [1/4] Loading compressor model from: {compressor_model_name}")
        self.compressor = AutoModelForCausalLM.from_pretrained(
            compressor_model_name, 
            dtype=dtype
        )
        print(f"  [2/4] Loading compressor tokenizer from: {compressor_model_name}")
        self.compressor_tokenizer = AutoTokenizer.from_pretrained(compressor_model_name)
        if self.compressor_tokenizer.pad_token is None:
            self.compressor_tokenizer.pad_token = self.compressor_tokenizer.eos_token
        
        # Add special memory tokens (like PCC does)
        print(f"  [2/4] Adding {n_mem_tokens} memory tokens to compressor tokenizer...")
        new_token_dict = {'additional_special_tokens': [f'<mem_{i}>' for i in range(n_mem_tokens)]}
        num_added_tokens = self.compressor_tokenizer.add_special_tokens(new_token_dict)
        self.compressor.resize_token_embeddings(len(self.compressor_tokenizer))
        self.mem_ids = [self.compressor_tokenizer.convert_tokens_to_ids(f'<mem_{i}>') for i in range(n_mem_tokens)]
        print(f"  [2/4] ✓ Compressor loaded and configured")
        
        # Apply LoRA if needed
        if use_lora:
            if lora_config is None:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=64,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
                )
            self.compressor = get_peft_model(self.compressor, lora_config)
        
        # Load decoder (frozen target model)
        print(f"  [3/4] Loading decoder model from: {decoder_model_name}")
        print(f"        (This may take a while for large models...)")
        
        # Validate decoder model path exists and contains necessary files
        decoder_path = Path(decoder_model_name)
        if not decoder_path.exists():
            raise FileNotFoundError(
                f"Decoder model path does not exist: {decoder_model_name}\n"
                f"Please provide a valid model path."
            )
        
        # Check for tokenizer files
        tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'tokenizer.model', 'spiece.model']
        has_tokenizer = any((decoder_path / f).exists() for f in tokenizer_files)
        if not has_tokenizer:
            print(f"        Warning: No standard tokenizer files found in {decoder_model_name}")
            print(f"        Attempting to load tokenizer anyway (may use remote files)...")
        
        self.decoder = AutoModelForCausalLM.from_pretrained(
            decoder_model_name,
            dtype=dtype
        )
        
        # Ensure tokenizer loads from the correct model path (not checkpoint directory)
        # Use trust_remote_code=True if needed for some models
        print(f"  [4/4] Loading decoder tokenizer from: {decoder_model_name}")
        
        # Check tokenizer config if it exists
        tokenizer_config_path = decoder_path / 'tokenizer_config.json'
        if tokenizer_config_path.exists():
            import json
            try:
                with open(tokenizer_config_path, 'r') as f:
                    tokenizer_config = json.load(f)
                    vocab_file = tokenizer_config.get('vocab_file')
                    if vocab_file is not None:
                        vocab_path = decoder_path / vocab_file
                        if not vocab_path.exists():
                            print(f"        Warning: vocab_file '{vocab_file}' specified in config but not found")
                            print(f"        Expected at: {vocab_path}")
            except Exception as e:
                print(f"        Warning: Could not read tokenizer config: {e}")
        
        try:
            # Try loading with explicit local_files_only=False to allow remote fallback
            self.decoder_tokenizer = AutoTokenizer.from_pretrained(
                decoder_model_name,
                trust_remote_code=True,
                local_files_only=False
            )
        except TypeError as e:
            # Handle the specific "not a string" error
            if "not a string" in str(e) or "vocab_file" in str(e).lower():
                print(f"        Error: Tokenizer vocab_file is not a valid string")
                print(f"        This usually means the tokenizer config is corrupted or incomplete")
                print(f"        Attempting to load from model name instead...")
                # Try loading by model name if it's a HuggingFace model identifier
                if '/' in decoder_model_name and not Path(decoder_model_name).exists():
                    # It might be a HuggingFace model ID
                    try:
                        self.decoder_tokenizer = AutoTokenizer.from_pretrained(
                            decoder_model_name,
                            trust_remote_code=True
                        )
                    except Exception as e3:
                        raise ValueError(
                            f"Failed to load tokenizer. The model path '{decoder_model_name}' "
                            f"may be incorrect or tokenizer files are missing/corrupted.\n"
                            f"Original error: {e}\n"
                            f"Fallback error: {e3}"
                        )
                else:
                    raise ValueError(
                        f"Tokenizer vocab_file is not a valid string. "
                        f"Please check the tokenizer configuration in: {decoder_model_name}\n"
                        f"Error: {e}"
                    )
            else:
                raise
        except Exception as e:
            print(f"        Warning: Failed to load tokenizer with trust_remote_code=True: {e}")
            print(f"        Retrying without trust_remote_code...")
            try:
                self.decoder_tokenizer = AutoTokenizer.from_pretrained(
                    decoder_model_name,
                    local_files_only=False
                )
            except Exception as e2:
                error_msg = str(e2)
                print(f"        Error: Failed to load tokenizer from {decoder_model_name}")
                print(f"        Error details: {error_msg}")
                print(f"        Please ensure this is a valid model path with tokenizer files.")
                print(f"        Expected tokenizer files: {', '.join(tokenizer_files)}")
                raise
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token
        
        # Freeze decoder
        print(f"  [4/4] Freezing decoder parameters...")
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.eval()
        
        # Config
        self.n_mem_tokens = n_mem_tokens
        self.compressor_hidden_size = self.compressor.config.hidden_size
        self.decoder_hidden_size = self.decoder.config.hidden_size
        
        print(f"  [4/4] Initializing projector...")
        self.projector = Projector(self.compressor_hidden_size, self.decoder_hidden_size)
        print(f"  ✓ SimpleCompressor initialization complete")
    
    def compress(self, input_ids, attention_mask):
        """
        Compress input using compressor model (following PCC approach, non-segmented version)
        Returns: compressed memory tokens (batch_size, n_mem_tokens, decoder_hidden_size)
        """
        # Append memory tokens to input (like PCC Compressor.forward does)
        # This allows the model to learn how to encode information in these special tokens
        batch_size = input_ids.size(0)
        mem_ids_tensor = torch.tensor(self.mem_ids, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
        input_ids_with_mem = torch.cat((input_ids, mem_ids_tensor), dim=1)
        
        # Create attention mask for the extended sequence (all ones for memory tokens)
        mem_attention = torch.ones((batch_size, self.n_mem_tokens), dtype=attention_mask.dtype, device=attention_mask.device)
        full_attention_mask = torch.cat((attention_mask, mem_attention), dim=1)
        
        # Forward through compressor with memory tokens (using autocast like PCC)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.compressor(
                input_ids=input_ids_with_mem,
                attention_mask=full_attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Extract hidden states at memory token positions (last n_mem_tokens)
        # This matches PCC: embedding = text_embedding.hidden_states[-1][:,-self.embed_len:,:]
        hidden_states = outputs.hidden_states[-1]  # (batch_size, seq_len + n_mem_tokens, compressor_hidden_size)
        compressed = hidden_states[:, -self.n_mem_tokens:, :]  # (batch_size, n_mem_tokens, compressor_hidden_size)
        
        # Use projector (converter) to convert from compressor dimension to decoder dimension
        compressed = self.projector(compressed)  # (batch_size, n_mem_tokens, decoder_hidden_size)
        
        return compressed
    
    def forward(self, compress_input_ids, compress_attention_mask, 
                decoder_input_ids, decoder_attention_mask, prompt_text=None):
        """
        Forward pass for AE task (non-segmented version of PCC)
        
        Format: [BOS] + [Memory] + [Target]
        
        Key insight: After Hugging Face's internal shift:
        - logits[i] predicts labels[i+1]
        - So logits at "last memory position" should predict Target[0]
        
        This means we need:
        - inputs_embeds = [BOS] + [Memory] + [Target] (full target, for teacher forcing)
        - labels = [-100] * (1 + n_mem) + [Target] (full target, to predict)
        """
        # Compress input (non-segmented: process entire sequence at once)
        compressed_memory = self.compress(compress_input_ids, compress_attention_mask)
        
        # Get BOS token embedding
        bos_token_id = self.decoder_tokenizer.bos_token_id
        if bos_token_id is None:
            bos_token_id = self.decoder_tokenizer.eos_token_id
        bos_embedding = self.decoder.get_input_embeddings()(
            torch.tensor([[bos_token_id]], device=compressed_memory.device)
        ).expand(compressed_memory.shape[0], -1, -1)
        
        # Get FULL target embeddings (not shifted - model needs to predict Target[0] from memory!)
        target_embeddings = self.decoder.get_input_embeddings()(decoder_input_ids)
        
        # Concatenate: [BOS] + [Memory] + [Target]
        # Note: No separate prompt - the memory itself signals reconstruction
        inputs_embeds = torch.cat([bos_embedding, compressed_memory, target_embeddings], dim=1)
        
        # Prepare attention mask: [1] + [1]*n_mem + [target_mask]
        bos_mask = torch.ones(
            compressed_memory.shape[0], 1,
            dtype=decoder_attention_mask.dtype,
            device=decoder_attention_mask.device
        )
        memory_mask = torch.ones(
            compressed_memory.shape[0], 
            self.n_mem_tokens, 
            dtype=decoder_attention_mask.dtype,
            device=decoder_attention_mask.device
        )
        full_attention_mask = torch.cat([bos_mask, memory_mask, decoder_attention_mask], dim=1)
        
        # Prepare labels: [-100] * (1 + n_mem) + [Target]
        # After Hugging Face shift:
        #   - shift_labels[n_mem] = labels[n_mem + 1] = Target[0]
        #   - shift_logits[n_mem] = logits from inputs_embeds[n_mem] = last memory token
        # This forces model to predict Target[0] from memory, not from teacher forcing!
        prefix_labels = torch.full(
            (compressed_memory.shape[0], self.n_mem_tokens + 1),  # BOS + Memory
            -100,
            dtype=decoder_input_ids.dtype,
            device=decoder_input_ids.device
        )
        
        # Target labels: FULL target (including Target[0]!)
        target_labels = decoder_input_ids.clone()
        target_labels[decoder_attention_mask == 0] = -100  # mask padding
        
        labels = torch.cat([prefix_labels, target_labels], dim=1)
        
        # Ensure compressor is in train mode, decoder in eval mode
        if not self.compressor.training:
            self.compressor.train()
        if self.decoder.training:
            self.decoder.eval()
        
        # Forward through decoder
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
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
    
    def generate(self, compress_input_ids, compress_attention_mask, max_new_tokens=50, **kwargs):
        """
        Generate text using compressed input
        """
        # Compress input
        compressed_memory = self.compress(compress_input_ids, compress_attention_mask)
        
        # Prepare attention mask for memory tokens
        memory_mask = torch.ones(
            compressed_memory.shape[0],
            self.n_mem_tokens,
            dtype=torch.long,
            device=compressed_memory.device
        )
        
        # Generate
        outputs = self.decoder.generate(
            inputs_embeds=compressed_memory,
            attention_mask=memory_mask,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
        return outputs
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        compressor_model_name: str = None,
        decoder_model_name: str = None,
        n_mem_tokens: int = None,
        use_lora: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device: str = 'cuda',
    ):
        """
        Load SimpleCompressor from a checkpoint directory.
        
        Args:
            checkpoint_path: Path to checkpoint directory (containing model.safetensors or pytorch_model.bin)
            compressor_model_name: Compressor model path (if None, will infer from checkpoint path)
            decoder_model_name: Decoder model path (if None, will infer from checkpoint path)
            n_mem_tokens: Number of memory tokens (if None, will try to infer from checkpoint)
            use_lora: Whether model uses LoRA
            dtype: Model dtype
            device: Device to load model on
            
        Returns:
            Loaded SimpleCompressor instance
        """
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
        
        # Infer model names from checkpoint path if not provided
        # Path format: {compressor}_to_{decoder}_mem{n}_len{l}_ds_4gpu
        if compressor_model_name is None or decoder_model_name is None:
            dir_name = checkpoint_path_obj.name
            if '_to_' in dir_name:
                parts = dir_name.split('_to_')
                compressor_name = parts[0]
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
                
                if compressor_model_name is None:
                    if compressor_name in model_paths:
                        compressor_model_name = model_paths[compressor_name]
                        print(f"Inferred compressor model: {compressor_name} -> {compressor_model_name}")
                    else:
                        raise ValueError(
                            f"Could not infer compressor model path. "
                            f"Inferred name: {compressor_name}, Known: {list(model_paths.keys())}"
                        )
                
                if decoder_model_name is None:
                    if decoder_name in model_paths:
                        decoder_model_name = model_paths[decoder_name]
                        print(f"Inferred decoder model: {decoder_name} -> {decoder_model_name}")
                    else:
                        raise ValueError(
                            f"Could not infer decoder model path. "
                            f"Inferred name: {decoder_name}, Known: {list(model_paths.keys())}"
                        )
            else:
                raise ValueError(
                    f"Could not infer model names from checkpoint path: {checkpoint_path}\n"
                    f"Path format should be: .../{{compressor}}_to_{{decoder}}_mem{{n}}_len{{l}}_ds_4gpu\n"
                    f"Please provide compressor_model_name and decoder_model_name explicitly."
                )
        
        # Infer n_mem_tokens from path if not provided
        if n_mem_tokens is None:
            dir_name = checkpoint_path_obj.name
            if '_mem' in dir_name:
                try:
                    # Extract number after _mem
                    mem_part = dir_name.split('_mem')[1].split('_')[0]
                    n_mem_tokens = int(mem_part)
                    print(f"Inferred n_mem_tokens: {n_mem_tokens}")
                except (ValueError, IndexError):
                    raise ValueError(
                        f"Could not infer n_mem_tokens from path: {checkpoint_path}\n"
                        f"Please provide n_mem_tokens explicitly."
                    )
            else:
                raise ValueError(
                    f"Could not infer n_mem_tokens from checkpoint path: {checkpoint_path}\n"
                    f"Please provide n_mem_tokens explicitly."
                )
        
        # Validate model paths exist and are directories (not checkpoint files)
        from pathlib import Path as PathLib
        compressor_path = PathLib(compressor_model_name)
        decoder_path = PathLib(decoder_model_name)
        
        if not compressor_path.exists():
            raise FileNotFoundError(
                f"Compressor model path does not exist: {compressor_model_name}\n"
                f"Please provide a valid model path (not a checkpoint directory)."
            )
        if not decoder_path.exists():
            raise FileNotFoundError(
                f"Decoder model path does not exist: {decoder_model_name}\n"
                f"Please provide a valid model path (not a checkpoint directory)."
            )
        
        # Check if paths look like checkpoint directories vs model directories
        # A checkpoint directory typically:
        # - Contains only model.safetensors or pytorch_model.bin (and maybe metrics.json, trainer_state.json)
        # - Does NOT contain config.json, tokenizer files
        # A model directory typically:
        # - Contains config.json, tokenizer files, AND model files
        
        def is_checkpoint_directory(path):
            """Check if a path looks like a checkpoint directory (not a full model directory)"""
            has_model_file = (path / 'model.safetensors').exists() or (path / 'pytorch_model.bin').exists()
            has_config = (path / 'config.json').exists()
            has_tokenizer = any((path / f).exists() for f in ['tokenizer.json', 'tokenizer_config.json', 'vocab.json'])
            
            # If it has model files but NOT config and tokenizer, it's likely a checkpoint
            # If it has all three, it's a full model directory
            return has_model_file and not (has_config and has_tokenizer)
        
        if is_checkpoint_directory(compressor_path):
            raise ValueError(
                f"Compressor model path appears to be a checkpoint directory: {compressor_model_name}\n"
                f"A checkpoint directory contains only model files, not config.json or tokenizer files.\n"
                f"Please provide the base model path (which contains config.json and tokenizer files).\n"
                f"Checkpoint path should be passed as 'checkpoint_path' parameter."
            )
        if is_checkpoint_directory(decoder_path):
            raise ValueError(
                f"Decoder model path appears to be a checkpoint directory: {decoder_model_name}\n"
                f"A checkpoint directory contains only model files, not config.json or tokenizer files.\n"
                f"Please provide the base model path (which contains config.json and tokenizer files)."
            )
        
        # Initialize model with base models
        print(f"\nInitializing SimpleCompressor...")
        print(f"  Compressor model: {compressor_model_name}")
        print(f"  Decoder model: {decoder_model_name}")
        print(f"  Memory tokens: {n_mem_tokens}")
        print(f"  This may take a few minutes to load the models...\n")
        
        model = cls(
            compressor_model_name=compressor_model_name,
            decoder_model_name=decoder_model_name,
            n_mem_tokens=n_mem_tokens,
            use_lora=use_lora,
            dtype=dtype
        )
        
        print(f"\nModels loaded. Now loading weights from checkpoint...")
        
        # Load weights from checkpoint
        checkpoint_path_safetensors = checkpoint_path_obj / 'model.safetensors'
        checkpoint_path_pytorch = checkpoint_path_obj / 'pytorch_model.bin'
        
        if checkpoint_path_safetensors.exists():
            print(f"Loading weights from safetensors: {checkpoint_path_safetensors}")
            if not SAFETENSORS_AVAILABLE:
                raise ImportError(
                    "safetensors library is required. Install with: pip install safetensors"
                )
            state_dict = safetensors_load(checkpoint_path_safetensors)
        elif checkpoint_path_pytorch.exists():
            print(f"Loading weights from pytorch_model.bin: {checkpoint_path_pytorch}")
            checkpoint = torch.load(checkpoint_path_pytorch, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            raise FileNotFoundError(
                f"Model checkpoint not found in: {checkpoint_path}\n"
                f"Expected either:\n"
                f"  - {checkpoint_path_safetensors}\n"
                f"  - {checkpoint_path_pytorch}"
            )
        
        # Filter state dict to only include compressor and projector weights
        # Decoder weights are not saved in checkpoint (frozen during training)
        # Tokenizer files are also not saved (loaded from original model path)
        print("Filtering state dict to load only compressor and projector weights...")
        model_dict = model.state_dict()
        filtered_dict = {}
        compressor_keys = []
        projector_keys = []
        skipped_keys = []
        
        for k, v in state_dict.items():
            # Only load compressor and projector weights
            if k.startswith('compressor.') or k.startswith('projector.'):
                if k in model_dict and v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
                    if k.startswith('compressor.'):
                        compressor_keys.append(k)
                    elif k.startswith('projector.'):
                        projector_keys.append(k)
                else:
                    skipped_keys.append(f"{k} (shape mismatch: {v.shape} vs {model_dict.get(k, 'missing')})")
            elif k.startswith('decoder.'):
                # Skip decoder weights (they are frozen and loaded from original model path)
                skipped_keys.append(f"{k} (decoder weights not loaded from checkpoint)")
            else:
                # Skip unknown keys
                skipped_keys.append(f"{k} (unknown key)")
        
        # Load filtered weights
        missing_keys, unexpected_keys = model.load_state_dict(filtered_dict, strict=False)
        
        # Filter out expected missing keys (decoder weights, etc.)
        expected_missing_prefixes = ['decoder.', 'compressor_tokenizer.', 'decoder_tokenizer.']
        unexpected_missing_keys = [
            key for key in missing_keys 
            if not any(key.startswith(prefix) for prefix in expected_missing_prefixes)
        ]
        
        print(f"✓ Loaded weights successfully:")
        print(f"  Compressor keys: {len(compressor_keys)}")
        print(f"  Projector keys: {len(projector_keys)}")
        print(f"  Total loaded: {len(filtered_dict)}/{len(state_dict)} keys")
        
        if skipped_keys and len(skipped_keys) <= 20:
            print(f"  Skipped keys from checkpoint: {len(skipped_keys)}")
            for key in skipped_keys[:10]:  # Show first 10 skipped keys
                print(f"    - {key}")
            if len(skipped_keys) > 10:
                print(f"    ... and {len(skipped_keys) - 10} more")
        
        # Only warn about unexpected missing keys (not decoder weights, which are loaded from original model)
        if unexpected_missing_keys:
            print(f"  Warning: {len(unexpected_missing_keys)} unexpected keys missing in checkpoint (using initialized values)")
            if len(unexpected_missing_keys) <= 10:
                for key in unexpected_missing_keys:
                    print(f"    - {key}")
        elif len(missing_keys) > len(unexpected_missing_keys):
            # Some keys were expected to be missing (e.g., decoder weights)
            expected_missing_count = len(missing_keys) - len(unexpected_missing_keys)
            print(f"  Note: {expected_missing_count} expected keys missing (decoder weights loaded from original model path)")
        
        if unexpected_keys:
            print(f"  Warning: {len(unexpected_keys)} unexpected keys in checkpoint (ignored)")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"    - {key}")
        
        # Move to device and set to eval mode
        model.to(device)
        model.eval()
        
        # Ensure Projector weights are in correct dtype
        if hasattr(model, 'projector'):
            for param in model.projector.parameters():
                if param.dtype != dtype:
                    param.data = param.data.to(dtype)
        
        return model