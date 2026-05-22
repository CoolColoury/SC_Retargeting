"""
ProjectorMLP: MLP-based Projector architecture for supervised transfer learning

This module defines the model architecture only. Training logic is in transfer.py.
Similar to converter/mlp_st_converter.py but simplified to just the model definition.
"""

import torch
import torch.nn as nn


class LlamaRMSNorm(nn.Module):
    """RMSNorm for Llama-style normalization"""
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


class ProjectorMLP(nn.Module):
    """
    MLP Projector for supervised transfer learning
    
    Architecture: Input -> LayerNorm -> Linear -> GELU -> Linear -> Output
    
    This is similar to SimpleCompressor's Projector but can be trained with
    supervised learning using paired source-target embeddings.
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
        return self.net(x)
    
    def print_parameters(self):
        """Print trainable parameter count"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"ProjectorMLP: {trainable_params:,} trainable / {total_params:,} total parameters")


def create_projector_mlp(src_dim, tgt_dim, hidden_layers=None, dropout=0.1):
    """
    Factory function to create ProjectorMLP
    
    Args:
        src_dim: Source embedding dimension
        tgt_dim: Target embedding dimension
        hidden_layers: List of hidden layer sizes
        dropout: Dropout rate
        
    Returns:
        ProjectorMLP instance
    """
    projector = ProjectorMLP(
        input_size=src_dim,
        output_size=tgt_dim,
        hidden_layers=hidden_layers,
        dropout=dropout
    )
    projector.print_parameters()
    return projector


if __name__ == "__main__":
    # Test the projector
    print("Testing ProjectorMLP...")
    
    # Create projector
    projector = create_projector_mlp(
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
