"""
HSTU: Hierarchical Sequential Transduction Unit
From "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"
https://arxiv.org/abs/2402.17152

Key differences from standard Transformer:
1. SiLU activation instead of softmax normalization (captures preference intensity)
2. Update gate U for gating mechanism
3. Relative attention bias with both position and temporal components
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class RotaryEmbedding(nn.Module):
    """
    旋转位置编码 (RoPE) 的核心实现。
    预计算余弦和正弦频率矩阵。
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 计算频率倒数
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # 预计算 max_seq_len 长度的 cos 和 sin，加速前向传播
        t = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) # [max_seq_len, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)           # [max_seq_len, dim]
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 返回当前序列长度对应的 cos 和 sin
        return (
            self.cos_cached[:seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:seq_len, ...].to(dtype=x.dtype),
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将特征维度的后半部分取负并与前半部分交换，用于 RoPE 的正交旋转"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 RoPE 应用到 Query 和 Key 上
    q, k: [B, H, L, d]
    cos, sin: [L, d]
    """
    # 增加维度以便广播: [L, d] -> [1, 1, L, d]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    # 公式: q * cos(theta) + rotate_half(q) * sin(theta)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class FeatureInteractionBlock(nn.Module):
    """
    带有【显式特征交叉】的深度交互融合网络。
    结合了 DCN-v2 (Deep & Cross Network) 的乘法交互机制与 GRN 门控。
    """
    def __init__(self, concat_dim: int, base_dim: int, dropout: float = 0.1):
        """
        Args:
            concat_dim: 拼接后的总维度 (e.g., 4 * base_dim)
            base_dim: 基础特征维度 (D)
            dropout: 门控层后的 dropout
        """
        super().__init__()
        self.concat_dim = concat_dim
        
        # 0. 初始化 LayerNorm
        self.pre_norm = nn.LayerNorm(concat_dim)
        
        # 🌟 1. 显式特征交叉投影 (Cross Projection)
        # 负责将输入打乱重组，以便与自身进行乘法交互
        self.cross_proj = nn.Linear(concat_dim, concat_dim)
        
        # 2. 深度非线性提取与门控 (Bottleneck)
        # 用于提取交叉后的高阶模式，并输出门控信号
        bottleneck_dim = 2 * base_dim 
        self.bottleneck = nn.Sequential(
            nn.Linear(concat_dim, bottleneck_dim),
            nn.SiLU(), 
            nn.Linear(bottleneck_dim, concat_dim) # 直接映射回 concat_dim 用于 Sigmoid
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, flattened_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flattened_x: 原始拼接后的向量 [..., concat_dim]
        Returns:
            深度交叉与融合后的向量 [..., concat_dim]
        """
        # 初始残差分支
        residual = flattened_x 
        
        # a. 标准化
        x = self.pre_norm(flattened_x)
        
        # === 🌟 核心改进：显式特征交叉 (Explicit Feature Interaction) ===
        # 原理: x ⊙ (Wx + b) 
        # 这里发生了真正的向量乘法。ID 维度的值会与映射后的 Sem 维度的值直接相乘，
        # 实现了推荐系统中极为重要的二阶特征交叉 (2nd-order Feature Crossing)。
        cross_out = x * self.cross_proj(x)
        
        # 将一阶原始特征与二阶交叉特征相加 (DCNv2 标准做法)
        cross_x = x + cross_out 
        # ==============================================================

        # b. 深度模式提取与门控信号生成
        # 模型根据交叉后的丰富特征，决定放行哪些信息
        gate_signals = torch.sigmoid(self.bottleneck(cross_x)) 
        
        # c. 门控过滤与残差相加
        # 用门控过滤交叉特征，再加上最原始的输入
        return residual + self.dropout(gate_signals * cross_x)
    
class SemanticFusionLayer(nn.Module):
    """
    语义 ID 展平层
    负责将 RQ-VAE 的 [..., 3] 语义 ID 映射并展平为 [..., 3 * embed_dim]
    """
    def __init__(self, num_codebooks: int, codebook_size: int, embed_dim: int):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        
        # 词表大小 = codebook数量 * 每个codebook的大小 + 1 (用于全局 Padding)
        self.vocab_size = num_codebooks * codebook_size + 1
        self.emb = nn.Embedding(self.vocab_size + 1000, embed_dim, padding_idx=0)

    def forward(self, semantic_ids: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            semantic_ids: [..., num_codebooks]
            token_type_ids: [..., num_codebooks]
        Returns:
            [..., num_codebooks * embed_dim]
        """
        valid_mask = (semantic_ids != 0).long()
        mapped_ids = (token_type_ids * self.codebook_size + semantic_ids + 1) * valid_mask
        
        # 获取各自的 Embedding: [..., num_codebooks, D]
        embs = self.emb(mapped_ids)
        
        # 🌟 核心修改：直接展平拼接，保持 3 * embed_dim 维度
        flattened_embs = embs.view(*embs.shape[:-2], self.num_codebooks * self.embed_dim)
        
        return flattened_embs
    
class HMSR_HSTU(nn.Module):
    def __init__(
        self,
        num_items: int,
        item_semantic_map: torch.Tensor, 
        max_seq_len: int = 50,
        embed_dim: int = 64, # 这里指的是单一特征的基础维度
        num_heads: int = 2,
        num_blocks: int = 2,
        dropout: float = 0.2,
        num_position_buckets: int = 32,
        num_time_buckets: int = 64,
        max_position_distance: int = 128,
        use_temporal_bias: bool = True,
        num_codebooks: int = 3,          
        codebook_size: int = 256,        
    ):
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_codebooks = num_codebooks
        self.use_temporal_bias = use_temporal_bias

        self.register_buffer("item_semantic_map", item_semantic_map)

        # 1. 传统 Item Embedding -> 输出维度 [..., embed_dim]
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        
        # 2. 语义 Semantic Embedding -> 输出维度 [..., 3 * embed_dim]
        self.semantic_fusion = SemanticFusionLayer(num_codebooks, codebook_size, embed_dim)

        # 🌟 核心修改 1：计算拼接后的总维度
        # 总维度 = 基础 ID 维度 (1倍) + 语义 ID 维度 (3倍) = 4 * embed_dim
        self.concat_dim = embed_dim + (num_codebooks * embed_dim)

        self.emb_dropout = nn.Dropout(dropout)

        self.fi = FeatureInteractionBlock(4*embed_dim, embed_dim, dropout)  # 🌟 新增：交互融合网络块


        # 🌟 核心修改 2：HSTU 块使用拼接后的总维度 (concat_dim)
        # 🌟 修改点：HSTU 块的参数传递适配 RoPE
        self.layers = nn.ModuleList([
            HSTULayer(
                embed_dim=self.concat_dim,  
                num_heads=num_heads, dropout=dropout,
                max_seq_len=max_seq_len, # <-- 传入最大序列长度给 RoPE
                num_time_buckets=num_time_buckets,
                use_temporal_bias=use_temporal_bias,
            )
            for _ in range(num_blocks)
        ])

        # 🌟 核心修改 3：最后的 LayerNorm 也要对应总维度
        self.final_norm = nn.LayerNorm(self.concat_dim)
        self._init_weights()

    # ... _init_weights 保持不变 ...
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

    def get_fused_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        动态提取并拼接： [传统ID嵌入 + 3个语义嵌入]
        """
        # [..., D]
        id_embs = self.item_embedding(item_ids) 
        
        sem_ids = self.item_semantic_map[item_ids] 
        type_ids = torch.arange(self.num_codebooks, device=item_ids.device).expand_as(sem_ids)
        
        # [..., 3 * D]
        sem_embs = self.semantic_fusion(sem_ids, type_ids) 
        
        # 🌟 核心修改：在最后一个维度拼接 -> [..., 4 * D]
        return torch.cat([id_embs, sem_embs], dim=-1)

    def forward(
        self,
        input_item_ids: torch.Tensor,       
        history_sid: torch.Tensor,          
        token_type_ids: torch.Tensor,       
        timestamps: Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None, 
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        B, L = input_item_ids.shape
        device = input_item_ids.device

        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        padding_mask = (input_item_ids == 0)

        # 1. Item ID 特征 [B, L, D]
        x_id = self.item_embedding(input_item_ids)
        
        # 2. Semantic ID 特征 [B, L, 3D]
        sem_ids_3d = history_sid.view(B, L, self.num_codebooks)
        type_ids_3d = token_type_ids.view(B, L, self.num_codebooks)
        x_sem = self.semantic_fusion(sem_ids_3d, type_ids_3d)

        # 3. 🌟 拼接融合 -> [B, L, 4D]
        x = self.emb_dropout(torch.cat([x_id, x_sem], dim=-1))
        
        x = self.fi(x)
        

        # 穿过 HSTU 层 (HSTU 现在内部维度是 4D)
        for layer in self.layers:
            x = layer(x, causal_mask, padding_mask, timestamps)
        
        x = self.final_norm(x) # [B, L, 4D]

        # 预测打分：用全量物品的【拼接特征】来算 Logits，维度完美匹配 4D @ 4D.T
        all_item_ids = torch.arange(self.num_items + 1, device=device)
        all_fused_embs = self.get_fused_item_embeddings(all_item_ids) # [V, 4D]
        
        

        logits = x @ all_fused_embs.T  # [B, L, V]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_items + 1),
                targets.view(-1),
                ignore_index=0
            )

        return logits, loss

    def forward_sampled_softmax(
        self,
        input_item_ids: torch.Tensor,
        history_sid: torch.Tensor,
        token_type_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None, 
        num_negatives: int = 128,
        temperature: float = 0.05,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        B, L = input_item_ids.shape
        device = input_item_ids.device

        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        padding_mask = (input_item_ids == 0)

        # 提取并拼接特征
        x_id = self.item_embedding(input_item_ids)
        sem_ids_3d = history_sid.view(B, L, self.num_codebooks)
        type_ids_3d = token_type_ids.view(B, L, self.num_codebooks)
        x_sem = self.semantic_fusion(sem_ids_3d, type_ids_3d)
        
        # 🌟 拼接
        x = self.emb_dropout(torch.cat([x_id, x_sem], dim=-1))

        for layer in self.layers:
            x = layer(x, causal_mask, padding_mask, timestamps)

        x = self.final_norm(x) # [B, L, 4D]
        
        all_item_ids = torch.arange(self.num_items + 1, device=device)
        all_fused_embs = self.get_fused_item_embeddings(all_item_ids)
        logits = x @ all_fused_embs.T

        loss = None
        if targets is not None:
            # 修改 view 的维度为 self.concat_dim
            x_flat = x.view(-1, self.concat_dim)
            targets_flat = targets.view(-1)

            valid_mask = targets_flat != 0
            x_valid = x_flat[valid_mask]
            targets_valid = targets_flat[valid_mask]

            if x_valid.size(0) > 0:
                pos_emb = self.get_fused_item_embeddings(targets_valid)

                neg_ids = torch.randint(1, self.num_items + 1, (num_negatives,), device=device)
                neg_emb = self.get_fused_item_embeddings(neg_ids)

                x_valid = F.normalize(x_valid, dim=-1)
                pos_emb = F.normalize(pos_emb, dim=-1)
                neg_emb = F.normalize(neg_emb, dim=-1)

                pos_logits = (x_valid * pos_emb).sum(dim=-1, keepdim=True)
                neg_logits = x_valid @ neg_emb.T
                all_logits = torch.cat([pos_logits, neg_logits], dim=-1) / temperature

                loss_targets = torch.zeros(x_valid.size(0), device=device, dtype=torch.long)
                loss = F.cross_entropy(all_logits, loss_targets)

        return logits, loss

# 注意：HSTULayer, RelativePositionBias, TemporalBias 保持你给出的原样即可，无需修改。

class HSTULayer(nn.Module):
    """
    带有 RoPE (旋转位置编码) 的 HSTU layer.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        max_seq_len: int,          # 🌟 替换了原有的 num_position_buckets 和 max_position_distance
        num_time_buckets: int,
        use_temporal_bias: bool,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_temporal_bias = use_temporal_bias

        assert embed_dim % num_heads == 0
        assert self.head_dim % 2 == 0, "RoPE requires head_dim to be even" # RoPE的限制条件

        # Pointwise projection: projects to 4 * embed_dim (for U, V, Q, K)
        self.projection = nn.Linear(embed_dim, 4 * embed_dim)

        # 🌟 引入旋转位置编码器
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

        # Temporal bias (保留，用于真实时间差)
        if use_temporal_bias:
            self.temporal_bias = TemporalBias(
                num_buckets=num_time_buckets,
                num_heads=num_heads,
            )

        self.attn_norm = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  
        causal_mask: torch.Tensor,  
        padding_mask: torch.Tensor,  
        timestamps: Optional[torch.Tensor] = None,  
    ) -> torch.Tensor:
        B, L, D = x.shape
        residual = x

        projected = F.silu(self.projection(x))  # [B, L, 4D]
        U, V, Q, K = projected.chunk(4, dim=-1)  # Each [B, L, D]

        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d]
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # === 🌟 应用旋转位置编码 (RoPE) ===
        cos, sin = self.rotary_emb(V, seq_len=L)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        # ==================================

        # === Spatial Aggregation ===
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 🌟 注：原有的 relative pos bias 偏置相加已被移除

        if self.use_temporal_bias and timestamps is not None:
            time_bias = self.temporal_bias(timestamps)
            scores = scores + time_bias

        attn_weights = F.silu(scores)

        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), 0.0)
        attn_weights = attn_weights.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), 0.0)

        attn_output = attn_weights @ V  # [B, H, L, d]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)  

        attn_output = self.attn_norm(attn_output)
        attn_output = attn_output * U  

        x = residual + self.dropout(attn_output)
        x = x + self.ffn(self.ffn_norm(x))

        return x

class RelativePositionBias(nn.Module):
    """
    Relative position bias using logarithmic bucketing (T5-style).

    Buckets relative positions into logarithmically spaced bins,
    allowing the model to generalize to longer sequences.
    """

    def __init__(self, num_buckets: int = 32, max_distance: int = 128, num_heads: int = 2):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads

        # Learnable bias for each bucket and head
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """
        Convert relative position to bucket index using logarithmic bucketing.

        For causal attention, we only care about positions where query >= key,
        so relative_position >= 0.
        """
        # We use half buckets for exact positions, half for log-spaced
        num_buckets = self.num_buckets
        max_distance = self.max_distance

        # Clamp to non-negative (causal)
        relative_position = torch.clamp(relative_position, min=0)

        # Half buckets for small distances (exact)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Log-spaced buckets for larger distances
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()

        relative_position_if_large = torch.clamp(relative_position_if_large, max=num_buckets - 1)

        bucket = torch.where(is_small, relative_position, relative_position_if_large)
        return bucket

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Compute relative position bias matrix.

        Returns:
            bias: [num_heads, seq_len, seq_len]
        """
        # Create position indices
        positions = torch.arange(seq_len, device=device)
        # relative_position[i, j] = i - j (query_pos - key_pos)
        relative_position = positions.unsqueeze(0) - positions.unsqueeze(1)  # [L, L]

        # Convert to buckets
        buckets = self._relative_position_bucket(relative_position)  # [L, L]

        # Look up bias values
        bias = self.relative_attention_bias(buckets)  # [L, L, H]
        bias = bias.permute(2, 0, 1)  # [H, L, L]

        return bias


class TemporalBias(nn.Module):
    """
    Temporal attention bias using logarithmic bucketing of time differences.

    Quantizes timestamp differences into log-spaced buckets,
    capturing both recent and long-term temporal patterns.
    """

    def __init__(self, num_buckets: int = 64, num_heads: int = 2):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads

        # Learnable bias for each bucket and head
        self.temporal_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _temporal_bucket(self, time_diff: torch.Tensor) -> torch.Tensor:
        """
        Convert time difference to bucket index.

        Uses formula: bucket = floor(log(max(1, |diff|)) / log_base)
        where log_base ≈ 0.301 (log10(2)) as in the paper
        """
        # Take absolute value and ensure minimum of 1
        abs_diff = torch.clamp(torch.abs(time_diff), min=1).float()

        # Log bucketing (using natural log, scaled)
        # Paper uses: floor(log(max(1, |diff|)) / 0.301)
        # We use a similar approach but cap at num_buckets - 1
        buckets = (torch.log(abs_diff) / 0.693).long()  # 0.693 = ln(2)

        buckets = torch.clamp(buckets, min=0, max=self.num_buckets - 1)
   
        return buckets

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal bias matrix.

        Args:
            timestamps: [B, L] unix timestamps

        Returns:
            bias: [B, num_heads, L, L]
        """
        B, L = timestamps.shape

        # Compute pairwise time differences
        # time_diff[i, j] = timestamps[i] - timestamps[j]
        time_diff = timestamps.unsqueeze(2) - timestamps.unsqueeze(1)  # [B, L, L]

        # Convert to buckets
        buckets = self._temporal_bucket(time_diff)  # [B, L, L]
        #print("Temporal buckets:", buckets)  # Debug: 输出桶的形状
        
        # Look up bias values
        bias = self.temporal_attention_bias(buckets)  # [B, L, L, H]
        bias = bias.permute(0, 3, 1, 2)  # [B, H, L, L]

        return bias



