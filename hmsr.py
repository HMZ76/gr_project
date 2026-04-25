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

class SemanticFusionLayer(nn.Module):
    """
    语义 ID 融合层
    负责将 RQ-VAE 展平的 [B, L*3] 的语义特征，还原、映射并融合为 [B, L, D] 的稠密向量。
    """
    def __init__(self, num_codebooks: int, codebook_size: int, embed_dim: int):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        
        # 词表大小 = codebook数量 * 每个codebook的大小 + 1 (用于全局 Padding)
        self.vocab_size = num_codebooks * codebook_size + 1
        self.emb = nn.Embedding(self.vocab_size+1000, embed_dim, padding_idx=0)
        
        # 融合投影 (可选：对相加后的语义向量进行一次非线性变换)
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, semantic_ids: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            semantic_ids: [..., 3]
            token_type_ids: [..., 3]
        """
        # 防止 padding 的 0 被映射到奇怪的区间
        valid_mask = (semantic_ids != 0).long()
        
        # 空间映射：把 3 个 codebook 拉平到一个 Vocab 空间中。+1 是为了给 padding 留出 0
        mapped_ids = (token_type_ids * self.codebook_size + semantic_ids + 1) * valid_mask
        
        # [..., 3, D]
        embs = self.emb(mapped_ids)
        
        # 在 Codebook 维度上求和融合 [..., 3, D] -> [..., D]
        summed_embs = embs.sum(dim=-2)
        
        return self.fusion_proj(summed_embs)


class HMSR_HSTU(nn.Module):
    """
    Hybrid Semantic & Item ID HSTU (HMSR-HSTU)
    """
    def __init__(
        self,
        num_items: int,
        item_semantic_map: torch.Tensor, # 🌟 新增：全量物品的 Semantic IDs 映射表 [num_items+1, 3]
        max_seq_len: int = 50,
        embed_dim: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
        dropout: float = 0.2,
        num_position_buckets: int = 32,
        num_time_buckets: int = 64,
        max_position_distance: int = 128,
        use_temporal_bias: bool = True,
        num_codebooks: int = 3,          # RQ-VAE codebook 数量
        codebook_size: int = 256,        # RQ-VAE codebook 字典大小
    ):
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.use_temporal_bias = use_temporal_bias
        self.num_codebooks = num_codebooks

        # 将全局语义映射表注册为 Buffer (不参与梯度更新，但随模型保存并在同设备上)
        # 确保它的 shape 是 [num_items+1, 3]
        self.register_buffer("item_semantic_map", item_semantic_map)

        # 1. 传统 Item Embedding
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        
        # 2. 语义 Semantic Embedding
        self.semantic_fusion = SemanticFusionLayer(num_codebooks, codebook_size, embed_dim)

        self.emb_dropout = nn.Dropout(dropout)

        # 3. HSTU Blocks
        self.layers = nn.ModuleList([
            HSTULayer(
                embed_dim=embed_dim, num_heads=num_heads, dropout=dropout,
                num_position_buckets=num_position_buckets, num_time_buckets=num_time_buckets,
                max_position_distance=max_position_distance, use_temporal_bias=use_temporal_bias,
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

    def get_fused_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        核心机制：动态提取任意 item_ids 的 [传统ID嵌入 + 语义嵌入]
        """
        # 1. 基础 Item ID Embedding
        id_embs = self.item_embedding(item_ids) # [..., D]
        
        # 2. 查表获取 Semantic IDs
        
        sem_ids = self.item_semantic_map[item_ids] # [..., 3]
        
        # 3. 动态生成 Token Type IDs (0, 1, 2)
        type_ids = torch.arange(self.num_codebooks, device=item_ids.device).expand_as(sem_ids)
        
        # 4. 获取语义 Embedding
        sem_embs = self.semantic_fusion(sem_ids, type_ids) # [..., D]
        
        # 🌟 强强联手：融合
        return id_embs + sem_embs

    def forward(
        self,
        input_item_ids: torch.Tensor,       # [B, L]
        history_sid: torch.Tensor,          # [B, L*3] (来自 hmsr_collate_fn)
        token_type_ids: torch.Tensor,       # [B, L*3] (来自 hmsr_collate_fn)
        timestamps: Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None, # 这里指 target_item_ids [B, L]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        B, L = input_item_ids.shape
        device = input_item_ids.device

        # Mask 生成
        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        padding_mask = (input_item_ids == 0)

        # === 🌟 历史特征双轨融合 ===
        # 1. 处理 Item ID
        x_id = self.item_embedding(input_item_ids)
        
        # 2. 处理 Semantic ID (将 [B, L*3] 转换为 [B, L, 3] 后送入语义层)
        sem_ids_3d = history_sid.view(B, L, self.num_codebooks)
        type_ids_3d = token_type_ids.view(B, L, self.num_codebooks)
        x_sem = self.semantic_fusion(sem_ids_3d, type_ids_3d)

        # 3. 相加融合
        x = self.emb_dropout(x_id + x_sem)

        # === 穿过 HSTU 层 ===
        for layer in self.layers:
            x = layer(x, causal_mask, padding_mask, timestamps)

        x = self.final_norm(x) # [B, L, D]

        # === 🌟 预测打分层 (Prediction) ===
        # 我们需要用全量物品的【融合特征】来算 Logits，而不是光光用 item_embedding
        all_item_ids = torch.arange(self.num_items + 1, device=device)
        all_fused_embs = self.get_fused_item_embeddings(all_item_ids) # [V, D]
        
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
        targets: Optional[torch.Tensor] = None, # [B, L]
        num_negatives: int = 128,
        temperature: float = 0.05,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        带负采样的训练模式 (针对混合语义架构进行了重构)
        """
        B, L = input_item_ids.shape
        device = input_item_ids.device

        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        padding_mask = (input_item_ids == 0)

        # 融合前向
        x_id = self.item_embedding(input_item_ids)
        sem_ids_3d = history_sid.view(B, L, self.num_codebooks)
        type_ids_3d = token_type_ids.view(B, L, self.num_codebooks)
        x_sem = self.semantic_fusion(sem_ids_3d, type_ids_3d)
        
        x = self.emb_dropout(x_id + x_sem)

        for layer in self.layers:
            x = layer(x, causal_mask, padding_mask, timestamps)

        x = self.final_norm(x) # [B, L, D]
        
        # 评估用 Full Logits
        all_item_ids = torch.arange(self.num_items + 1, device=device)
        all_fused_embs = self.get_fused_item_embeddings(all_item_ids)
        logits = x @ all_fused_embs.T

        loss = None
        if targets is not None:
            x_flat = x.view(-1, self.embed_dim)
            targets_flat = targets.view(-1)

            valid_mask = targets_flat != 0
            x_valid = x_flat[valid_mask]
            targets_valid = targets_flat[valid_mask]

            if x_valid.size(0) > 0:
                # 🌟 获取正样本的融合 Embedding
                pos_emb = self.get_fused_item_embeddings(targets_valid)

                # 🌟 获取负样本的融合 Embedding (采样后动态拼接语义)
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
    Single HSTU layer.

    Structure:
        1. Pointwise Projection: X -> SiLU(Linear(X)) -> split to U, V, Q, K
        2. Spatial Aggregation: SiLU(QK^T + RAB) @ V
        3. Pointwise Transformation: Norm(Attention) ⊙ U -> FFN
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        num_position_buckets: int,
        num_time_buckets: int,
        max_position_distance: int,
        use_temporal_bias: bool,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_temporal_bias = use_temporal_bias

        assert embed_dim % num_heads == 0

        # Pointwise projection: projects to 4 * embed_dim (for U, V, Q, K)
        self.projection = nn.Linear(embed_dim, 4 * embed_dim)

        # Relative attention bias (position-based, shared across heads)
        self.position_bias = RelativePositionBias(
            num_buckets=num_position_buckets,
            max_distance=max_position_distance,
            num_heads=num_heads,
        )

        # Temporal bias (optional)
        if use_temporal_bias:
            self.temporal_bias = TemporalBias(
                num_buckets=num_time_buckets,
                num_heads=num_heads,
            )

        # Layer norm for attention output
        self.attn_norm = nn.LayerNorm(embed_dim)

        # FFN (pointwise transformation)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Final layer norm
        self.ffn_norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # [B, L, D]
        causal_mask: torch.Tensor,  # [L, L]
        padding_mask: torch.Tensor,  # [B, L]
        timestamps: Optional[torch.Tensor] = None,  # [B, L]
    ) -> torch.Tensor:
        B, L, D = x.shape
        residual = x

        # === Pointwise Projection ===
        projected = F.silu(self.projection(x))  # [B, L, 4D]
        U, V, Q, K = projected.chunk(4, dim=-1)  # Each [B, L, D]

        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d]
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # === Spatial Aggregation ===
        # 🚨 必须修复 1: 增加缩放因子，防止梯度爆炸！
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        pos_bias = self.position_bias(L, x.device)  
        scores = scores + pos_bias.unsqueeze(0)

        if self.use_temporal_bias and timestamps is not None:
            time_bias = self.temporal_bias(timestamps)
            scores = scores + time_bias

        # 🚨 必须修复 2: 先过 SiLU 激活函数！
        attn_weights = F.silu(scores)

        # 🚨 必须修复 3: 激活后再将无效位置（Padding和未来信息）精确 Mask 为 0.0！
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), 0.0)
        attn_weights = attn_weights.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), 0.0)

        attn_output = attn_weights @ V  # [B, H, L, d]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)  

        # === Pointwise Transformation ===
        attn_output = self.attn_norm(attn_output)
        attn_output = attn_output * U  # Element-wise gating

        # Residual connection
        x = residual + self.dropout(attn_output)

        # FFN with residual
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

        # Look up bias values
        bias = self.temporal_attention_bias(buckets)  # [B, L, L, H]
        bias = bias.permute(0, 3, 1, 2)  # [B, H, L, L]

        return bias



