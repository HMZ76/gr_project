import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import collections

# ==========================================
# 极简 PyTorch K-Means 聚类函数
# ==========================================
def kmeans_pytorch(x, num_clusters, num_iters=50):
    N, D = x.shape
    indices = torch.randperm(N)[:num_clusters]
    centroids = x[indices].clone()
    
    for _ in range(num_iters):
        dist = torch.cdist(x, centroids)
        labels = dist.argmin(dim=1)
        for k in range(num_clusters):
            mask = (labels == k)
            if mask.any():
                centroids[k] = x[mask].mean(dim=0)
    return centroids

# ==========================================
# 1. 定义 RQ-VAE 核心组件
# ==========================================

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, dim, num_codebooks, codebook_size):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, dim)) for _ in range(num_codebooks)
        ])
        
    def forward(self, z):
        residual = z
        quantized_out = 0
        indices = []
        
        for i in range(self.num_codebooks):
            codebook = self.codebooks[i]
            dist = (
                residual.pow(2).sum(1, keepdim=True) 
                - 2 * residual @ codebook.t() 
                + codebook.pow(2).sum(1, keepdim=True).t()
            )
            idx = dist.argmin(dim=-1)
            indices.append(idx)
            
            z_q = F.embedding(idx, codebook)
            residual = residual - z_q
            quantized_out = quantized_out + z_q
            
        quantized_st = z + (quantized_out - z).detach()
        return quantized_st, indices

    @torch.no_grad()
    def init_codebooks_with_kmeans(self, z, num_iters=20):
        print("开始 K-Means 码本初始化...")
        residual = z.clone()
        for i in range(self.num_codebooks):
            print(f"  正在初始化第 {i+1}/{self.num_codebooks} 层码本...")
            centroids = kmeans_pytorch(residual, self.codebook_size, num_iters=num_iters)
            self.codebooks[i].data.copy_(centroids)
            
            dist = torch.cdist(residual, centroids)
            idx = dist.argmin(dim=-1)
            z_q = F.embedding(idx, centroids)
            residual = residual - z_q
        print("K-Means 码本初始化完成！\n")

class RQVAE(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=[512, 256, 32], num_codebooks=4, codebook_size=256):
        """
        支持多层 hidden_dims 配置
        :param hidden_dims: List[int], 例如 [512, 256, 32]，最后一维 32 是传入 RQ 的维度
        """
        super().__init__()
        
        # 容错处理：如果用户传了 int，自动转为 list
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
            
        # 1. 动态构建 Encoder (逐层降维)
        enc_layers = []
        in_d = input_dim
        for out_d in hidden_dims[:-1]:
            enc_layers.append(nn.Linear(in_d, out_d))
            enc_layers.append(nn.BatchNorm1d(out_d)) # 加速收敛
            enc_layers.append(nn.ReLU())
            in_d = out_d
        # 最后一层映射到 Latent 空间（不加 ReLU，保留负值表达能力）
        enc_layers.append(nn.Linear(in_d, hidden_dims[-1]))
        self.encoder = nn.Sequential(*enc_layers)
        
        # 2. RQ-VAE 量化层，维度使用 hidden_dims 的最后一层
        latent_dim = hidden_dims[-1]
        self.rq = ResidualVectorQuantizer(latent_dim, num_codebooks, codebook_size)
        
        # 3. 动态构建 Decoder (逐层反向升维)
        dec_layers = []
        reversed_dims = list(reversed(hidden_dims)) # 例如 [32, 256, 512]
        in_d = reversed_dims[0]
        
        for out_d in reversed_dims[1:]:
            dec_layers.append(nn.Linear(in_d, out_d))
            dec_layers.append(nn.ReLU())
            in_d = out_d
        # 最后一层映射回原维度 input_dim
        dec_layers.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        z_q, indices = self.rq(z)
        x_recon = self.decoder(z_q)
        return x_recon, z, z_q, indices

# ==========================================
# 2. 评估逻辑定义
# ==========================================

def get_collision_rate(model, data_tensor, device, codebook_size=256):
    model.eval()
    all_indices = []
    with torch.no_grad():
        batch_size = 1024
        for i in range(0, data_tensor.size(0), batch_size):
            batch_x = data_tensor[i:i+batch_size].to(device)
            z = model.encoder(batch_x)
            _, indices = model.rq(z)
            all_indices.append(torch.stack(indices, dim=1).cpu())
    
    full_indices = torch.cat(all_indices, dim=0).numpy()
    num_items = full_indices.shape[0]
    
    unique_codes = len(set(tuple(row) for row in full_indices))
    collision_rate = (1 - unique_codes / num_items) * 100
    
    # 计算每一层码本的利用率
    layer_usages = []
    for i in range(full_indices.shape[1]):
        layer_usage = len(np.unique(full_indices[:, i])) / codebook_size * 100
        layer_usages.append(layer_usage)
    
    return collision_rate, layer_usages

if __name__ == "__main__":

    
    # 替换为你自己的路径
    import pandas as pd
    df = pd.read_parquet("../genrec/dataset/amazon/processed/beauty/item_emb_sentence-t5-xl.parquet")
    array = df['embedding'].values
    array = np.array([x.tolist() for x in array])
    array = array[1:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_tensor = torch.from_numpy(array).float()
    train_loader = DataLoader(TensorDataset(data_tensor), batch_size=128, shuffle=True)

    hidden_dims_list = [512, 256, 128, 64, 32] 
    model = RQVAE(
        input_dim=768, 
        hidden_dims=hidden_dims_list, 
        num_codebooks=3, 
        codebook_size=256
    ).to(device)

    # K-Means 初始化
    model.eval()
    with torch.no_grad():
        z_initial = model.encoder(data_tensor.to(device))
        model.rq.init_codebooks_with_kmeans(z_initial, num_iters=20)

    # 设置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # ---------------------------------------------------------
    # 【新增】定义 Warm-up 学习率调度器
    # ---------------------------------------------------------
    warmup_epochs = 100
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 线性预热：从 0 增加到 1.0 的倍率
            return float(epoch + 1) / float(max(1, warmup_epochs))
        # 预热结束后，保持基础学习率 1.0 倍
        return 1.0 
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # ---------------------------------------------------------

    epochs = 1000
    print(f"开始训练，设备: {device} | Encoder 结构: 768 -> {' -> '.join(map(str, hidden_dims_list))}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            x = batch[0].to(device)
            x_recon, z, z_q, _ = model(x)
            
            recon_loss = criterion(x_recon, x)
            commit_loss = criterion(z_q.detach(), z)
            loss = recon_loss + commit_loss 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 【新增】每个 epoch 结束后更新学习率
        scheduler.step()
        
        c_rate, layer_usages = get_collision_rate(model, data_tensor, device)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            # 格式化所有层的利用率显示
            usage_str = " | ".join([f"码本{i+1}利用率: {usage:5.1f}%" for i, usage in enumerate(layer_usages)])
            print(f"Epoch [{epoch+1:4d}/{epochs}] | LR: {current_lr:.6f} | Loss: {total_loss/len(train_loader):.9f} | "
                  f"碰撞率: {c_rate:5.2f}% | {usage_str}")
        
        if epoch > 0 and epoch % 50 == 0:
            print("重新初始化码本...")
            model.eval()
            with torch.no_grad():
                z_initial = model.encoder(data_tensor.to(device))
                model.rq.init_codebooks_with_kmeans(z_initial, num_iters=20)
        
    print("训练完成！")

    # ==========================================
    # 4. 保存模型权重
    # ==========================================
    save_path = "rqvae_best_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"模型权重已保存至: {save_path}")