import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 19
    n_heads: int = 64
    n_kv_heads: int = 4
    head_dim: int = 64
    vocab_size: int = 262144  # 262k
    max_seq_len: int = 131072 # 128k
    
    # Dense block (First block)
    dense_ffn_hidden_dim: int = 8192
    
    # MoE blocks
    n_experts: int = 128
    n_active_experts: int = 6
    moe_expert_hidden_dim: int = 1024
    
    norm_eps: float = 1e-5
    use_grad_checkpoint: bool = False

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    # Reshape to complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Apply rotation
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SwiGLU(nn.Module):
    """FeedForward (SwiGLU) module as depicted in the diagram"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False) # goes to SiLU
        self.w3 = nn.Linear(dim, hidden_dim, bias=False) # bypasses SiLU
        self.w2 = nn.Linear(hidden_dim, dim, bias=False) # Final output linear layer

    def forward(self, x):
        # (SiLU(x * W1) * (x * W3)) * W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class GroupedQueryAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        
        # QK-Norm explicitly shown in diagram
        self.q_norm = RMSNorm(args.head_dim)
        self.k_norm = RMSNorm(args.head_dim)

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape
        
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply QK-Norm before RoPE
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Grouped-Query Attention repeat K and V
        xk = torch.repeat_interleave(xk, self.n_rep, dim=2)
        xv = torch.repeat_interleave(xv, self.n_rep, dim=2)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # FlashAttention: fused scale + causal-mask + softmax + matmul in one
        # CUDA kernel. Avoids materialising the full [seqlen, seqlen] score
        # matrix in global memory. is_causal=True replaces the manual mask.
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class MoELayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_experts = args.n_experts
        self.n_active = args.n_active_experts
        
        # Router
        self.gate = nn.Linear(args.dim, args.n_experts, bias=False)
        
        # 128 Routed Experts
        self.experts = nn.ModuleList([
            SwiGLU(args.dim, args.moe_expert_hidden_dim) for _ in range(args.n_experts)
        ])
        
        # 1 Shared Expert (always active per token)
        self.shared_expert = SwiGLU(args.dim, args.moe_expert_hidden_dim)

    def forward(self, x):
        bsz, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)

        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(routing_weights, self.n_active, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_output = torch.zeros_like(x_flat)

        # One pass per expert instead of n_active × n_experts passes.
        # For each expert we gather *all* tokens assigned to it (across every
        # active slot), run a single batched forward, then scatter-add the
        # weighted results back — no per-iteration GPU synchronisation.
        for expert_id in range(self.n_experts):
            # selected_experts: [n_tokens, n_active]
            expert_mask = (selected_experts == expert_id)   # [n_tokens, n_active]
            if not expert_mask.any():
                continue
            # token_indices: which tokens; slot_indices: which active slot matched
            token_indices, slot_indices = expert_mask.nonzero(as_tuple=True)
            weights = routing_weights[token_indices, slot_indices].unsqueeze(-1)

            expert_out = self.experts[expert_id](x_flat[token_indices])   # [k, dim]
            final_output.scatter_add_(
                0,
                token_indices.unsqueeze(-1).expand_as(expert_out),
                expert_out * weights,
            )

        shared_out = self.shared_expert(x_flat)
        return (final_output + shared_out).view(bsz, seq_len, dim)

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        
        self.attention = GroupedQueryAttention(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        # First block uses dense FFN instead of MoE
        if layer_id == 0:
            self.ffn = SwiGLU(args.dim, args.dense_ffn_hidden_dim)
        else:
            self.ffn = MoELayer(args)

    def forward(self, x, freqs_cis):
        # Attention with residual
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        # FFN/MoE with residual
        out = h + self.ffn(self.ffn_norm(h))
        return out

class CustomMoEModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        
        # Token embedding layer
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        
        # 19 Transformer blocks
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(args, layer_id))
            
        # Final RMSNorm
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        # Linear output layer
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Precompute RoPE frequencies (up to max context length 128k)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(args.head_dim, args.max_seq_len),
            persistent=False
        )

    def forward(self, tokens):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        
        # Get RoPE frequencies for the current sequence length
        freqs_cis = self.freqs_cis[:seqlen]
        
        for layer in self.layers:
            if self.args.use_grad_checkpoint:
                # Recompute activations during backward instead of storing them.
                # use_reentrant=False is required for correct behaviour with
                # kwargs and avoids a PyTorch deprecation warning.
                h = checkpoint(layer, h, freqs_cis, use_reentrant=False)
            else:
                h = layer(h, freqs_cis)
            
        h = self.norm(h)
        logits = self.output(h)
        return logits

# --- Example Usage ---
if __name__ == "__main__":
    # Initialize config strictly matching the image
    config = ModelArgs()
    
    # Instantiate model
    model = CustomMoEModel(config)
    print(f"Model successfully instantiated with {config.n_layers} layers.")