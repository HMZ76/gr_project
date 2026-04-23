import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class HSTU(nn.Module):
    def __init__(
        self,
        num_users: int = 10000,
        num_items: int = 20000,
        max_seq_len: int = 50,
        embed_dim: int = 32,
        num_heads: int = 2,
        num_blocks: int = 2,
        dropout: float = 0.2,
        num_position_buckets: int = 32,
        num_time_buckets: int = 64,
        max_position_distance: int = 128,
        use_temporal_bias: bool = True,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.use_temporal_bias = use_temporal_bias

        # User ID & Item ID Embeddings
        self.user_embedding = nn.Embedding(num_users + 1, embed_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            HSTULayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                num_position_buckets=num_position_buckets,
                num_time_buckets=num_time_buckets,
                max_position_distance=max_position_distance,
                use_temporal_bias=use_temporal_bias,
            )
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        user_ids: torch.Tensor,      # [B]
        input_ids: torch.Tensor,     # [B, L]
        timestamps: Optional[torch.Tensor] = None,  # [B, L]
        targets: Optional[torch.Tensor] = None, 
        temperature: float = 0.05, 
        l2_norm: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        B, L = input_ids.shape
        device = input_ids.device

        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        padding_mask = (input_ids == 0)

        # Fusion: Item + User
        x = self.item_embedding(input_ids)
        u_emb = self.user_embedding(user_ids)
        x = x + u_emb.unsqueeze(1)
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x, causal_mask, padding_mask, timestamps)

        x = self.final_norm(x)

        all_item_embeddings = self.item_embedding.weight
        if l2_norm:
            x_normalized = F.normalize(x, dim=-1)
            item_weights = F.normalize(all_item_embeddings, dim=-1)
            logits = (x_normalized @ item_weights.T) / temperature
        else:
            logits = x @ all_item_embeddings.T  

        loss = None
        if targets is not None:
            if targets.dim() == 1 or (targets.dim() == 2 and targets.shape[1] == 1):
                last_logits = logits[:, -1, :]  
                loss = F.cross_entropy(last_logits, targets.view(-1), ignore_index=0)
            else:
                loss = F.cross_entropy(logits.view(-1, self.num_items + 1), targets.view(-1), ignore_index=0)

        return logits, loss

    def forward_sampled_softmax(
        self,
        user_ids: torch.Tensor,
        input_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        num_negatives: int = 128,
        temperature: float = 0.05,
        l2_norm: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        B, L = input_ids.shape
        device = input_ids.device

        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        padding_mask = (input_ids == 0)

        x = self.item_embedding(input_ids)
        u_emb = self.user_embedding(user_ids)
        x = x + u_emb.unsqueeze(1)
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x, causal_mask, padding_mask, timestamps)

        x = self.final_norm(x)  

        logits, loss = None, None
        if targets is not None:
            x_flat = x[:, -1, :] if targets.dim() == 1 or (targets.dim() == 2 and targets.shape[1] == 1) else x.view(-1, self.embed_dim)
            targets_flat = targets.view(-1)
            valid_mask = targets_flat != 0
            x_valid, targets_valid = x_flat[valid_mask], targets_flat[valid_mask]  

            if x_valid.size(0) > 0:
                pos_emb = self.item_embedding(targets_valid)  
                neg_ids = torch.randint(1, self.num_items + 1, (num_negatives,), device=device)
                neg_emb = self.item_embedding(neg_ids)        

                if l2_norm:
                    x_valid = F.normalize(x_valid, dim=-1)
                    pos_emb = F.normalize(pos_emb, dim=-1)
                    neg_emb = F.normalize(neg_emb, dim=-1)

                pos_logits = (x_valid * pos_emb).sum(dim=-1, keepdim=True)  
                neg_logits = x_valid @ neg_emb.T  
                all_logits = torch.cat([pos_logits, neg_logits], dim=-1) / temperature

                loss_targets = torch.zeros(x_valid.size(0), device=device, dtype=torch.long)
                loss = F.cross_entropy(all_logits, loss_targets)

        return logits, loss

class HSTULayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, num_position_buckets: int, num_time_buckets: int, max_position_distance: int, use_temporal_bias: bool):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_temporal_bias = use_temporal_bias
        
        self.projection = nn.Linear(embed_dim, 4 * embed_dim)
        self.position_bias = RelativePositionBias(num_buckets=num_position_buckets, max_distance=max_position_distance, num_heads=num_heads)
        if use_temporal_bias:
            self.temporal_bias = TemporalBias(num_buckets=num_time_buckets, num_heads=num_heads)
            
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(4 * embed_dim, embed_dim), nn.Dropout(dropout))
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor, padding_mask: torch.Tensor, timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        residual = x
        projected = F.silu(self.projection(x))
        U, V, Q, K = projected.chunk(4, dim=-1)
        
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = Q @ K.transpose(-2, -1)
        scores = scores + self.position_bias(L, x.device).unsqueeze(0)
        
        if self.use_temporal_bias and timestamps is not None:
            scores = scores + self.temporal_bias(timestamps)
            
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
        scores = scores.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn_weights = F.silu(scores)
        
        attn_output = (attn_weights @ V).transpose(1, 2).contiguous().view(B, L, D)
        attn_output = self.attn_norm(attn_output) * U
        
        x = residual + self.dropout(attn_output)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class RelativePositionBias(nn.Module):
    def __init__(self, num_buckets: int = 32, max_distance: int = 128, num_heads: int = 2):
        super().__init__()
        self.num_buckets, self.max_distance = num_buckets, max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        relative_position = torch.clamp(relative_position, min=0)
        max_exact = self.num_buckets // 2
        is_small = relative_position < max_exact
        rel_pos_large = max_exact + (torch.log(relative_position.float() / max_exact) / math.log(self.max_distance / max_exact) * (self.num_buckets - max_exact)).long()
        return torch.where(is_small, relative_position, torch.clamp(rel_pos_large, max=self.num_buckets - 1))

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        buckets = self._relative_position_bucket(positions.unsqueeze(0) - positions.unsqueeze(1))
        return self.relative_attention_bias(buckets).permute(2, 0, 1)

class TemporalBias(nn.Module):
    def __init__(self, num_buckets: int = 64, num_heads: int = 2):
        super().__init__()
        self.num_buckets = num_buckets
        self.temporal_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _temporal_bucket(self, time_diff: torch.Tensor) -> torch.Tensor:
        abs_diff = torch.clamp(torch.abs(time_diff), min=1).float()
        buckets = (torch.log(abs_diff) / 0.693).long()
        return torch.clamp(buckets, min=0, max=self.num_buckets - 1)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        time_diff = timestamps.unsqueeze(2) - timestamps.unsqueeze(1)
        return self.temporal_attention_bias(self._temporal_bucket(time_diff)).permute(0, 3, 1, 2)