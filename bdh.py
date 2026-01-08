"""
Baby Dragon Hatchling (BDH) implementation.
Based on: https://github.com/pathwaycom/bdh
Paper: "The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain"
"""
import dataclasses
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Causal self-attention with RoPE (Rotary Position Embeddings)."""
    
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        
        # RoPE frequencies
        self.register_buffer(
            "rope_freqs",
            1.0 / (10000 ** (torch.arange(0, n_embd // n_head, 2).float() / (n_embd // n_head)))
        )
        
        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )
    
    def apply_rope(self, x):
        """Apply rotary position embeddings."""
        B, nh, T, hs = x.shape
        
        # Generate position indices
        pos = torch.arange(T, device=x.device).unsqueeze(1)
        
        # Compute angles
        freqs = self.rope_freqs[:hs // 2]
        angles = pos * freqs  # (T, hs/2)
        
        # Split into even/odd dimensions
        x_even = x[..., 0::2]  # (B, nh, T, hs/2)
        x_odd = x[..., 1::2]   # (B, nh, T, hs/2)
        
        # Apply rotation
        cos_a = torch.cos(angles).unsqueeze(0).unsqueeze(0)  # (1, 1, T, hs/2)
        sin_a = torch.sin(angles).unsqueeze(0).unsqueeze(0)  # (1, 1, T, hs/2)
        
        x_rotated_even = x_even * cos_a - x_odd * sin_a
        x_rotated_odd = x_even * sin_a + x_odd * cos_a
        
        # Interleave back
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        return x_rotated.flatten(-2)
    
    def forward(self, Q, K, V):
        """
        Args:
            Q, K, V: (B, T, n_embd) query, key, value tensors
        Returns:
            output: (B, T, n_embd) attention output
        """
        B, T, C = Q.shape
        
        # Reshape to multi-head: (B, T, n_head, head_size)
        Q = Q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        K = K.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        V = V.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
        # Apply RoPE
        QR = self.apply_rope(Q)
        KR = self.apply_rope(K)
        
        # Compute attention scores (no softmax - emergent attention)
        scores = (QR @ KR.transpose(-2, -1)) / math.sqrt(QR.size(-1))  # (B, nh, T, T)
        
        # Apply causal mask
        scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # Attention weights (no softmax in base BDH - using tanh for stability)
        att = torch.tanh(scores)
        att = F.dropout(att, p=self.dropout, training=self.training)
        
        # Apply attention to values
        output = att @ V  # (B, nh, T, hs)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        
        return output


class BDHBlock(nn.Module):
    """BDH transformer block with Hebbian-like sparse activations."""
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.sparse_dim = config.n_embd * config.mlp_internal_dim_multiplier
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        # Sparse encoder/decoder (for Hebbian-like activations)
        self.encoder = nn.Linear(config.n_embd, self.sparse_dim, bias=False)
        self.encoder_v = nn.Linear(config.n_embd, self.sparse_dim, bias=False)
        self.decoder = nn.Linear(self.sparse_dim, config.n_embd, bias=False)
        
        # Attention module
        self.attn = CausalSelfAttention(
            config.n_embd, 
            config.n_head, 
            config.dropout,
            config.block_size
        )
        
        self.dropout = config.dropout
    
    def forward(self, x):
        """
        Args:
            x: (B, T, n_embd) input tensor
        Returns:
            output: (B, T, n_embd) block output
        """
        # Hebbian working memory simulation
        # 1. Project to sparse space
        x_sparse = F.relu(self.encoder(self.ln1(x)))  # (B, T, sparse_dim)
        
        # 2. Attention in sparse space (queries and keys are sparse activations)
        yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)  # Attend with sparse Q,K but dense V
        
        # 3. Create value-based sparse activations
        y_sparse = F.relu(self.encoder_v(yKV))  # (B, T, sparse_dim)
        
        # 4. Hebbian-like multiplication (synaptic strengthening)
        xy_sparse = x_sparse * y_sparse  # Element-wise product
        
        # 5. Project back to dense space
        output = self.decoder(xy_sparse)  # (B, T, n_embd)
        
        # 6. Residual connection
        x = x + F.dropout(output, p=self.dropout, training=self.training)
        
        # 7. Feedforward (standard)
        x = x + F.dropout(self.ln2(x), p=self.dropout, training=self.training)
        
        return x


class BDH(nn.Module):
    """Baby Dragon Hatchling language model."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embedding (byte-level)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([BDHBlock(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        """
        Args:
            idx: (B, T) tensor of token indices
            targets: (B, T) tensor of target indices (optional)
        Returns:
            logits: (B, T, vocab_size) prediction scores
            loss: scalar cross-entropy loss (if targets provided)
        """
        B, T = idx.shape
        
        # Token embeddings
        x = self.token_embedding(idx)  # (B, T, n_embd)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    def get_embeddings(self, idx):
        """
        Extract embeddings for classification tasks.
        
        Args:
            idx: (B, T) tensor of token indices
        Returns:
            embeddings: (B, T, n_embd) final layer representations
        """
        B, T = idx.shape
        
        # Token embeddings
        x = self.token_embedding(idx)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        return x
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: (B, T) initial context
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling (None = no filtering)
        Returns:
            idx: (B, T + max_new_tokens) generated sequence
        """
        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Get last token logits
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
