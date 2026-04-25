import torch
import torch.nn as nn

torch.manual_seed(42)

# 构造一个形状为 [B, L, D] = [2, 3, 4] 的序列张量
# 2 个样本，序列长度为 3，特征维度为 4
B, L, D = 2, 3, 4
x = torch.randn(B, L, D)

print("=== 1. Layer Normalization (LayerNorm) ===")
# 官方 API
ln = nn.LayerNorm(normalized_shape=D)
out_ln_api = ln(x)

# 💡 手动底层实现：只沿着特征维度 D (dim=-1) 求均值和方差
# 每个样本、每个时间步的特征都拥有自己独立的均值和方差
mean_ln = x.mean(dim=-1, keepdim=True)  # 形状: [2, 3, 1]
var_ln = x.var(dim=-1, keepdim=True, unbiased=False)
out_ln_manual = (x - mean_ln) / torch.sqrt(var_ln + 1e-5)
# 乘以可学习参数 gamma，加上 beta (这里初始设为 1 和 0)
out_ln_manual = out_ln_manual * ln.weight + ln.bias

print("LayerNorm 官方与手动结果是否一致:", torch.allclose(out_ln_api, out_ln_manual, atol=1e-5))
print("LayerNorm Mean 形状:", mean_ln.shape)


print("\n=== 2. Batch Normalization (BatchNorm) ===")
# 注意工程细节：PyTorch 的 BatchNorm 期望特征通道在第 1 维，即 [B, D, L]
bn = nn.BatchNorm1d(num_features=D)
x_bn_format = x.transpose(1, 2)  # 转换形状为 [2, 4, 3]
out_bn_api = bn(x_bn_format).transpose(1, 2) # 计算完再转回 [2, 3, 4]

# 💡 手动底层实现：沿着 Batch 维度 B 和 序列维度 L 求均值和方差
# 同一个特征维度 D 上的所有元素，共享同一个均值和方差
# 对于 [B, L, D] 张量，就是在 dim=0 和 dim=1 上进行聚合
mean_bn = x.mean(dim=(0, 1), keepdim=True)  # 形状: [1, 1, 4]
var_bn = x.var(dim=(0, 1), keepdim=True, unbiased=False)
out_bn_manual = (x - mean_bn) / torch.sqrt(var_bn + 1e-5)
out_bn_manual = out_bn_manual * bn.weight + bn.bias

print("BatchNorm 官方与手动结果是否一致:", torch.allclose(out_bn_api, out_bn_manual, atol=1e-5))
print("BatchNorm Mean 形状:", mean_bn.shape)