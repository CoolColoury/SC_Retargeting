"""
ProjectorE2E: End-to-end Projector architecture for transfer learning

This module defines the model architecture only. Training logic is in transfer.py.
Similar to converter/e2e_train.py but simplified to just the model definition.
"""

import torch
import torch.nn as nn

from decoder_prefix_tokens import decoder_bos_token_id


class LlamaRMSNorm(nn.Module):
    """RMSNorm for Llama-style normalization"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_state):
        # Ensure weight is float32 (projector needs float32)
        if self.weight.dtype != torch.float32:
            self.weight.data = self.weight.data.to(torch.float32)
        
        # Normalize in float32
        input_dtype = hidden_state.dtype
        hidden_state = hidden_state.to(torch.float32)
        variance = hidden_state.pow(2).mean(-1, keepdim=True)
        hidden_state = hidden_state * torch.rsqrt(variance + self.variance_epsilon)
        
        # Multiply with weight (both in float32), then convert output
        output = self.weight * hidden_state
        return output.to(input_dtype)


class ProjectorE2E(nn.Module):
    """
    End-to-end Projector for transfer learning
    
    Architecture: Input -> LayerNorm -> MLP -> Output
    
    This projector is designed to be trained end-to-end with a target decoder model.
    The training loss comes from the decoder's reconstruction loss.
    
    Differences from ProjectorMLP:
    - Trained with reconstruction loss (not supervised loss)
    - Optimized end-to-end with decoder forward pass
    - Memory vectors are converted in the context of decoder usage
    """
    def __init__(self, input_size, output_size, hidden_layers=None, dropout=0.1):
        """
        Args:
            input_size: Source dimension
            output_size: Target dimension
            hidden_layers: List of hidden layer sizes (default: [2048])
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [2048]
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        
        # Build MLP layers
        layers = []
        prev_dim = input_size
        
        # Add input normalization
        layers.append(LlamaRMSNorm(input_size))
        
        # Add hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=True))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Add output layer with normalization
        layers.append(nn.Linear(prev_dim, output_size, bias=True))
        layers.append(nn.LayerNorm(output_size, elementwise_affine=True))
        
        self.net = nn.Sequential(*layers)
        
        self._init_weights()
        
        # Ensure all parameters are float32 (projector needs float32 for training)
        # This includes Linear layers, LayerNorm, and RMSNorm
        for param in self.parameters():
            if param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
    
    def _init_weights(self):
        """Initialize weights with small random values"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (..., input_size)
            
        Returns:
            Output tensor of shape (..., output_size)
        """
        # Ensure input is float32
        x = x.to(torch.float32)
        
        # Ensure all module parameters are float32
        # (DeepSpeed might convert them to bfloat16, so we need to convert back)
        # Check if any parameter is not float32 and convert all if needed
        needs_conversion = False
        for module in self.net.modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, LlamaRMSNorm)):
                for param in module.parameters():
                    if param.dtype != torch.float32:
                        needs_conversion = True
                        break
                if needs_conversion:
                    break
        
        if needs_conversion:
            # Convert all parameters back to float32
            for module in self.net.modules():
                if isinstance(module, (nn.Linear, nn.LayerNorm, LlamaRMSNorm)):
                    for param in module.parameters():
                        if param.dtype != torch.float32:
                            param.data = param.data.to(torch.float32)
        
        return self.net(x)
    
    def print_parameters(self):
        """Print trainable parameter count"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"ProjectorE2E: {trainable_params:,} trainable / {total_params:,} total parameters")


class E2EWrapper(nn.Module):
    """
    Wrapper that combines ProjectorE2E with a target decoder model for end-to-end training.
    
    This is used during training to compute reconstruction loss.
    During inference, only the projector is needed.
    """
    def __init__(self, projector, decoder_model, decoder_tokenizer=None):
        """
        Args:
            projector: ProjectorE2E instance
            decoder_model: Target decoder model (will be frozen)
            decoder_tokenizer: Target tokenizer (required for correct Qwen prefix ids)
        """
        super().__init__()
        self.projector = projector
        self.decoder = decoder_model
        self.decoder_tokenizer = decoder_tokenizer
        self._bos_token_id = None
        
        # Freeze decoder
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.eval()
    
    def forward(self, src_memory, target_input_ids, target_attention_mask):
        """
        Forward pass for training
        
        Args:
            src_memory: Source memory tensor (batch_size, n_mem_tokens, src_dim)
            target_input_ids: Target text input IDs (batch_size, seq_len)
            target_attention_mask: Target attention mask (batch_size, seq_len)
            
        Returns:
            loss: Reconstruction loss
            logits: Decoder logits
        """
        # Get decoder embedding layer to determine dtype
        decoder_embeddings = self.decoder.get_input_embeddings()
        target_dtype = decoder_embeddings.weight.dtype
        
        # Project source memory to target dimension
        # Projector works in float32, so convert input to float32 first
        src_memory_float = src_memory.to(torch.float32)
        projected_memory = self.projector(src_memory_float)
        
        # Convert projected memory to match decoder dtype
        projected_memory = projected_memory.to(dtype=target_dtype)
        
        if self._bos_token_id is None:
            if self.decoder_tokenizer is None:
                raise ValueError(
                    "E2EWrapper requires decoder_tokenizer for prefix token alignment "
                    "(pass tgt_tokenizer when constructing the wrapper)."
                )
            self._bos_token_id = decoder_bos_token_id(
                self.decoder_tokenizer, self.decoder.config
            )

        bos_embedding = decoder_embeddings(
            torch.tensor([[self._bos_token_id]], device=src_memory.device)
        ).expand(src_memory.shape[0], -1, -1)
        
        # Get target embeddings
        target_embeddings = decoder_embeddings(target_input_ids)
        
        # Concatenate: [BOS] + [Projected Memory] + [Target]
        inputs_embeds = torch.cat([bos_embedding, projected_memory, target_embeddings], dim=1)
        
        # Prepare attention mask
        n_mem_tokens = src_memory.shape[1]
        bos_mask = torch.ones(src_memory.shape[0], 1, dtype=target_attention_mask.dtype, device=src_memory.device)
        memory_mask = torch.ones(src_memory.shape[0], n_mem_tokens, dtype=target_attention_mask.dtype, device=src_memory.device)
        full_attention_mask = torch.cat([bos_mask, memory_mask, target_attention_mask], dim=1)
        
        # Prepare labels: [-100] * (1 + n_mem) + [Target]
        prefix_labels = torch.full(
            (src_memory.shape[0], n_mem_tokens + 1),
            -100,
            dtype=target_input_ids.dtype,
            device=src_memory.device
        )
        target_labels = target_input_ids.clone()
        target_labels[target_attention_mask == 0] = -100
        labels = torch.cat([prefix_labels, target_labels], dim=1)
        
        # Forward through decoder
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs.loss, outputs.logits


def create_projector_e2e(src_dim, tgt_dim, hidden_layers=None, dropout=0.1):
    """
    Factory function to create ProjectorE2E
    
    Args:
        src_dim: Source embedding dimension
        tgt_dim: Target embedding dimension
        hidden_layers: List of hidden layer sizes
        dropout: Dropout rate
        
    Returns:
        ProjectorE2E instance
    """
    projector = ProjectorE2E(
        input_size=src_dim,
        output_size=tgt_dim,
        hidden_layers=hidden_layers,
        dropout=dropout
    )
    projector.print_parameters()
    return projector


if __name__ == "__main__":
    # Test the projector
    print("Testing ProjectorE2E...")
    
    # Create projector
    projector = create_projector_e2e(
        src_dim=768,
        tgt_dim=1024,
        hidden_layers=[2048],
        dropout=0.1
    )
    
    # Test forward pass
    batch_size = 4
    n_mem_tokens = 32
    x = torch.randn(batch_size, n_mem_tokens, 768)
    
    print(f"\nInput shape: {x.shape}")
    y = projector(x)
    print(f"Output shape: {y.shape}")
    
    assert y.shape == (batch_size, n_mem_tokens, 1024), f"Expected (4, 32, 1024), got {y.shape}"
    print("\n✓ Test passed!")
