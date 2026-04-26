import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
# 假设你把上一版整合好的模型代码保存为了 tiger.py
from tiger import Tiger 

from dataset import TigerDataset, tiger_collate_fn

def evaluate(model, dataloader, device, k=10, valid_item_ids=None):
    """
    针对 TIGER (生成式推荐) 的正确评估函数
    """
    model.eval()  # 切入评估模式
    total_recall = 0.0
    total_ndcg = 0.0
    num_samples = 0

    # 使用 inference_mode 可以进一步降低 Beam Search 的显存消耗
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=f"Evaluating (Top-{k})", leave=False):
            batch = {key: val.to(device) for key, val in batch.items()}
            
            B = batch["user_input_ids"].size(0)
            
            # 1. 使用 Beam Search 生成 Top-K 的候选序列
            output = model.generate(
                user_input_ids=batch["user_input_ids"],
                item_input_ids=batch["item_input_ids"],
                token_type_ids=batch["token_type_ids"],
                seq_mask=batch["seq_mask"],
                n_top_k_candidates=k,
                valid_item_ids=valid_item_ids, 
                use_trie=(valid_item_ids is not None) # 如果有合法 ID 库，强制开启 Trie 约束
            )
            
            generated_sem_ids = output.sem_ids  # (B, K, sem_id_dim)
            target_sem_ids = batch["target_input_ids"]  # (B, sem_id_dim)
            
            # 2. 序列匹配判定 (3 个 Code 必须全量匹配才算对)
            matches = (generated_sem_ids == target_sem_ids.unsqueeze(1)).all(dim=2) # 形状: (B, K)
            
            # 3. 计算 Recall 和 NDCG
            for i in range(B):
                match_positions = matches[i].nonzero(as_tuple=True)[0]
                
                if len(match_positions) > 0:
                    rank = match_positions[0].item()
                    total_recall += 1.0
                    total_ndcg += 1.0 / math.log2(rank + 2.0) 
                    
            num_samples += B

    avg_recall = total_recall / num_samples if num_samples > 0 else 0.0
    avg_ndcg = total_ndcg / num_samples if num_samples > 0 else 0.0
    
    return avg_recall, avg_ndcg

def main():
    # ========================
    # 1. 配置与设备
    # ========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ========================
    # 2. 数据加载
    # ========================
    train_dataset = TigerDataset(
        root="../genrec/dataset/amazon",
        rqvae_path="rqvae_best_model.pth",
        train_test_split='train',
        device=device
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=tiger_collate_fn,
        num_workers=0  # 避免 DataLoader 中使用多进程导致 CUDA context 问题
    )

    val_dataset = TigerDataset(
        root="../genrec/dataset/amazon",
        rqvae_path="rqvae_best_model.pth",
        train_test_split='valid',
        device=device
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=tiger_collate_fn,
        num_workers=0
    )

    test_dataset = TigerDataset(
        root="../genrec/dataset/amazon",
        rqvae_path="rqvae_best_model.pth",
        train_test_split='test',
        device=device
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=tiger_collate_fn,
        num_workers=0
    )

    # ========================
    # 3. 模型初始化
    # ========================
    model = Tiger(
        embedding_dim=512,
        attn_dim=384,
        dropout=0.2,
        num_heads=8,
        n_layers=2,
        num_item_embeddings=256,   # 对应 RQ-VAE codebook_size
        num_user_embeddings=10000, # 增加用户嵌入数量以避免索引越界
        sem_id_dim=3               # 对应 RQ-VAE num_codebooks
    ).to(device)

    # ========================
    # 4. 优化器与训练设置
    # ========================
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    num_epochs = 100

    # ========================
    # 5. 训练循环
    # ========================
    for epoch in range(num_epochs):
        # 【关键修复】：确保每个 epoch 开始前，模型都处于 train 模式
        # 因为 evaluate 函数中会调用 model.eval()，如果不切回来，会导致后续 Epoch 没有 Dropout 且无法更新 Norm
        model.train() 
        
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 掩码容错处理
            if "seq_mask" in batch:
                mask_max = batch["seq_mask"].max().item()
                mask_min = batch["seq_mask"].min().item()
                if mask_max > 1.0 or mask_min < 0.0:
                    batch["seq_mask"] = torch.clamp(batch["seq_mask"], 0.0, 1.0)

            optimizer.zero_grad()
            output = model(
                user_input_ids=batch["user_input_ids"],
                item_input_ids=batch["item_input_ids"],
                token_type_ids=batch["token_type_ids"],
                target_input_ids=batch["target_input_ids"],
                target_token_type_ids=batch["target_token_type_ids"],
                seq_mask=batch["seq_mask"]
            )

            loss = output.loss
            if loss is not None:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{num_epochs}] | Avg Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] | No valid batches.")

        # ========================
        # 6. 每个 epoch 后评估
        # ========================
        # 注: 如果你有 valid_item_ids (全局合法的 Semantic IDs 集合)，请在这里传入
        recall_5, ndcg_5 = evaluate(model, val_dataloader, device, k=5, valid_item_ids=None)
        recall_10, ndcg_10 = evaluate(model, val_dataloader, device, k=10, valid_item_ids=None)
        test_recall_5, test_ndcg_5 = evaluate(model, test_dataloader, device, k=5, valid_item_ids=None)
        test_recall_10, test_ndcg_10 = evaluate(model, test_dataloader, device, k=10, valid_item_ids=None)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Recall@5: {recall_5:.4f} | NDCG@5: {ndcg_5:.4f} | Recall@10: {recall_10:.4f} | NDCG@10: {ndcg_10:.4f}")
        print(f"测试集评估结果 | Recall@5: {test_recall_5:.4f} | NDCG@5: {test_ndcg_5:.4f} | Recall@10: {test_recall_10:.4f} | NDCG@10: {test_ndcg_10:.4f}")
        print("-" * 80)

    # ========================
    # 7. 保存模型
    # ========================
    torch.save(model.state_dict(), "tiger_model_weights.pth")
    print("TIGER 模型权重已保存至 tiger_model_weights.pth")

    # =========================
    # 8. 测试模型
    # =========================
    model.load_state_dict(torch.load("tiger_model_weights.pth"))
    

if __name__ == "__main__":
    main()