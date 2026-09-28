"""Hand-built heterogeneous hypergraph transformer encoder.

Implements the node-to-hyperedge attention + hyperedge-to-node propagation
math from DESIGN.md's Structural Encoding section directly (no GNN library —
this is the actual message-passing math, not a call to a pre-built layer),
with two deliberate additions beyond that pseudo-math:

1. Attention is masked to each hyperedge's actual members. The formula as
   written in DESIGN.md — softmax((H^T X W_Q)(X W_K)^T / sqrt(d))(X W_V) —
   is ambiguous about this: taken completely literally, every hyperedge query
   attends to *every* node in the graph, using H only to build the initial
   query, which is not hypergraph message passing, it's full attention with
   extra steps. Masking the attention logits to each hyperedge's members
   (standard practice in real hypergraph transformer papers, e.g. HEGEL) is
   what actually makes this a *hypergraph* layer rather than a dense one.
2. Multi-head attention + LayerNorm — standard transformer training
   stabilizers, not in the design doc's math, but necessary for this to train
   at all rather than diverge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphAttentionLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.w_o = nn.Linear(dim, dim)
        self.w_r = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def _split_heads(self, t: torch.Tensor, seq_len: int) -> torch.Tensor:
        return t.view(seq_len, self.n_heads, self.head_dim).transpose(0, 1)

    def forward(self, x: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        """x: [N, dim] node features. incidence: [N, M] binary, dense."""
        n_nodes, n_edges = incidence.shape
        if n_edges == 0:
            return x

        # node -> hyperedge: hyperedge query starts as the mean of its
        # member nodes' features, then attends (masked to members only) over
        # all nodes to build a richer hyperedge representation.
        degree = incidence.sum(dim=0, keepdim=True).clamp(min=1)
        hyperedge_init = (incidence / degree).T @ x  # [M, dim]

        q = self._split_heads(self.w_q(hyperedge_init), n_edges)  # [heads, M, hd]
        k = self._split_heads(self.w_k(x), n_nodes)  # [heads, N, hd]
        v = self._split_heads(self.w_v(x), n_nodes)  # [heads, N, hd]

        attn_logits = q @ k.transpose(-2, -1) / (self.head_dim**0.5)  # [heads, M, N]
        mask = (incidence.T == 0).unsqueeze(0)  # [1, M, N] — True where not a member
        attn_logits = attn_logits.masked_fill(mask, float("-inf"))
        attn = attn_logits.softmax(dim=-1)
        # a hyperedge with zero members (e.g. after masking held-out edges
        # for validation) has an all -inf row -> softmax gives NaN (0/0);
        # zero those rows out instead so an empty hyperedge just contributes
        # nothing, rather than NaN-poisoning the whole forward pass
        empty_hyperedge = mask.all(dim=-1, keepdim=True)  # [1, M, 1]
        attn = torch.where(empty_hyperedge, torch.zeros_like(attn), attn)

        hyperedge_repr = (attn @ v).transpose(0, 1).reshape(n_edges, self.dim)
        hyperedge_repr = self.w_o(hyperedge_repr)  # [M, dim]

        # hyperedge -> node: each node sums the representations of every
        # hyperedge it belongs to, plus a residual node-level transform.
        node_update = incidence @ hyperedge_repr  # [N, dim]
        out = self.norm1(node_update + self.w_r(x))
        out = F.gelu(out)
        return self.norm2(out + x)


class HypergraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [HypergraphAttentionLayer(hidden_dim, n_heads) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, incidence)
        return x
