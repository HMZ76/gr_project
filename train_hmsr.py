import os
# 禁用 Tokenizer 的底层并行，防止与 DataLoader 的多进程产生冲突
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

# 🌟 确保从你的模型文件中导入新的 HMSR_HSTU
from hmsr import HMSR_HSTU  
# 🌟 引入配套的双轨 Collate Functions
from dataset import AmazonDataset, hmsr_collate_fn, hmsr_eval_collate_fn


def setup_logger(log_file):
    """Setup logger for training."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


def evaluate(model, dataloader, device, k_list=None):
    """
    Evaluate model on validation/test set with ranking metrics for multiple K values.
    Optimized to compute top-K only once per batch.
    """
    if k_list is None:
        k_list = [5, 10]
    
    max_k = max(k_list)
    model.eval()
    
    total_metrics = {k: {'recall': 0.0, 'ndcg': 0.0} for k in k_list}
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # 🌟 使用对齐的新字段名
            input_item_ids = batch['input_item_ids'].to(device)
            targets = batch['target_item_ids'].to(device) # Eval 时这是一个标量 [B]
            timestamps = batch['timestamps'].to(device)
            history_sid = batch['history_sid'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            
            # Forward pass (显式 kwargs 传参最安全)
            logits, _ = model(
                input_item_ids=input_item_ids,
                history_sid=history_sid,
                token_type_ids=token_type_ids,
                timestamps=timestamps,
                targets=None # 评估时不计算 Loss
            )
            
            if logits is not None:
                # Handle 3D logits from HSTU output (B, L, V)
                if logits.dim() == 3:
                    logits = logits[:, -1, :]  # (B, V) - prediction for last item
                
                # Mask padding index so it's never recommended
                logits[:, 0] = float('-inf')
                
                # Get Top-max_K indices
                _, topk_indices = torch.topk(logits, max_k, dim=-1)  # (B, max_k)
                
                # Compute hits matrix (B, max_k)
                hit_matrix = (topk_indices == targets.unsqueeze(1))
                
                batch_size = targets.size(0)
                num_samples += batch_size
                
                # Compute metrics for each K
                for k in k_list:
                    hits_k = hit_matrix[:, :k] # (B, k)
                    
                    # Recall@K
                    recall = hits_k.sum().item()
                    
                    # NDCG@K
                    hits_idx = hits_k.nonzero(as_tuple=True) 
                    ranks = hits_idx[1] 
                    weights = 1.0 / torch.log2(ranks.float() + 2.0)
                    ndcg = weights.sum().item()
                    
                    total_metrics[k]['recall'] += recall
                    total_metrics[k]['ndcg'] += ndcg
    
    avg_loss = float('inf')  # Evaluation phase doesn't compute CE loss
    avg_metrics = {}
    for k in k_list:
        avg_metrics[k] = {
            'recall': total_metrics[k]['recall'] / num_samples if num_samples > 0 else 0.0,
            'ndcg': total_metrics[k]['ndcg'] / num_samples if num_samples > 0 else 0.0
        }
    
    return avg_loss, avg_metrics


def train_epoch(model, dataloader, optimizer, scheduler, device, use_sampled_softmax=False):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    num_samples = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
        
        # 🌟 使用对齐的新字段名
        input_item_ids = batch['input_item_ids'].to(device)
        targets = batch['target_item_ids'].to(device) # 训练时这是一个序列 [B, L]
        timestamps = batch['timestamps'].to(device)
        history_sid = batch['history_sid'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)

        if use_sampled_softmax:
            _, loss = model.forward_sampled_softmax(
                input_item_ids=input_item_ids,
                history_sid=history_sid,
                token_type_ids=token_type_ids,
                timestamps=timestamps,
                targets=targets
            )
        else:
            _, loss = model(
                input_item_ids=input_item_ids,
                history_sid=history_sid,
                token_type_ids=token_type_ids,
                timestamps=timestamps,
                targets=targets
            )
        
        if loss is not None:
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item() * input_item_ids.size(0)
            num_samples += input_item_ids.size(0)
    
    avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train HMSR-HSTU model')
    parser.add_argument('--dataset', type=str, default='beauty', help='Dataset name')
    parser.add_argument('--root', type=str, default='../genrec/dataset/amazon', help='Dataset root path')
    parser.add_argument('--use_sampled_softmax', action='store_true', help='Use sampled softmax loss')
    parser.add_argument('--embed_dim', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--num_blocks', type=int, default=2, help='Number of HSTU blocks')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--max_seq_len', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('--num_position_buckets', type=int, default=32, help='Number of position buckets')
    parser.add_argument('--num_time_buckets', type=int, default=64, help='Number of time buckets')
    parser.add_argument('--max_position_distance', type=int, default=128, help='Max position distance')
    parser.add_argument('--use_temporal_bias', default=True, help='Use temporal attention bias')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--eval_k', type=int, default=10, help='K for Recall@K and NDCG@K')
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.checkpoint_dir, 'train_hstu.log'))
    logger.info(f"Training arguments: {args}")
    
    logger.info("Loading datasets...")
    # 使用修改后的 TigerDataset/AmazonDataset 均可，只要内部生成了 semantic_indices
    train_dataset = AmazonDataset(root=args.root, split=args.dataset, train_test_split="train", max_seq_len=args.max_seq_len)
    test_dataset = AmazonDataset(root=args.root, split=args.dataset, train_test_split="valid", max_seq_len=args.max_seq_len)
    valid_dataset  = AmazonDataset(root=args.root, split=args.dataset, train_test_split="test",  max_seq_len=args.max_seq_len)
    
    num_items = train_dataset.num_items
    logger.info(f"Number of items: {num_items}")
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Valid samples: {len(valid_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # 🌟 严格区分 Train 和 Eval 的 Collate Function
    collate_train = lambda x: hmsr_collate_fn(x, max_seq_len=args.max_seq_len)
    collate_eval  = lambda x: hmsr_eval_collate_fn(x, max_seq_len=args.max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_train, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_eval,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_eval,  num_workers=4, pin_memory=True)
    
    logger.info("Initializing HMSR-HSTU model...")
    # 🌟 将全局物品语义表转换为 Tensor 喂给模型
    # 🌟 将全局物品语义表转换为 Tensor
    item_semantic_map = torch.tensor(train_dataset.semantic_indices, dtype=torch.long)
    
    # 🚑 致命越界修复：检测并自动补齐 Padding 行
    if item_semantic_map.shape[0] == num_items:
        logger.info("⚠️ 检测到 semantic_map 缺少第 0 行 (Padding)，正在自动补零对齐...")
        pad_row = torch.zeros(1, 3, dtype=torch.long)
        item_semantic_map = torch.cat([pad_row, item_semantic_map], dim=0)
    elif item_semantic_map.shape[0] < num_items:
        raise ValueError(f"🚨 严重异常: 语义表行数 {item_semantic_map.shape[0]} 远小于 num_items {num_items}，请删除 dataset/amazon/processed 下的 parquet 缓存重试！")
    else:
        logger.info("✅ semantic_map 维度校验通过！")

    logger.info(f"最终 item_semantic_map 维度: {item_semantic_map.shape}") # 预期应该是 [num_items + 1, 3]
    
    model = HMSR_HSTU(
        num_items=num_items, 
        item_semantic_map=item_semantic_map, # 注入灵魂特征！
        max_seq_len=args.max_seq_len, 
        embed_dim=args.embed_dim, 
        num_heads=args.num_heads, 
        num_blocks=args.num_blocks, 
        dropout=args.dropout,
        num_position_buckets=args.num_position_buckets,
        num_time_buckets=args.num_time_buckets,
        max_position_distance=args.max_position_distance,
        use_temporal_bias=args.use_temporal_bias,
    ).to(args.device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_valid_ndcg = -float('inf')
    patience_counter = 0
    
    logger.info("Starting training...")
    eval_k_list = sorted(list(set([5, args.eval_k])))
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, None, args.device, use_sampled_softmax=args.use_sampled_softmax)
        
        # Evaluate on validation set during training
        _, valid_metrics = evaluate(model, valid_loader, args.device, k_list=eval_k_list)
        
        valid_recall_k = valid_metrics[args.eval_k]['recall']
        valid_ndcg_k = valid_metrics[args.eval_k]['ndcg']
        
        log_msg = f"Epoch {epoch+1:02d}/{args.epochs} | Train Loss: {train_loss:.4f}"
        for k in eval_k_list:
            log_msg += f" | Val Recall@{k}: {valid_metrics[k]['recall']:.4f} | Val NDCG@{k}: {valid_metrics[k]['ndcg']:.4f}"
        logger.info(log_msg)
        
        # Early Stopping Logic based on original eval_k NDCG
        if valid_ndcg_k > best_valid_ndcg:
            best_valid_ndcg = valid_ndcg_k
            patience_counter = 0
            
            checkpoint_path = os.path.join(args.checkpoint_dir, f'hmsr_hstu_{args.dataset}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_ndcg': valid_ndcg_k,
                'args': args
            }, checkpoint_path)
            logger.info(f"  🌟 Saved new best model with Val NDCG@{args.eval_k}: {valid_ndcg_k:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs (No improvement for {args.patience} epochs).")
                break
    
    # -----------------------------------------
    # Final Testing Phase
    # -----------------------------------------
    logger.info("Loading best model for Final Testing...")
    checkpoint_path = os.path.join(args.checkpoint_dir, f'hmsr_hstu_{args.dataset}_best.pth')
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, test_metrics = evaluate(model, test_loader, args.device, k_list=eval_k_list)
    
    test_msg = "🏆 Final Test Results:"
    for k in eval_k_list:
        test_msg += f"\n  - Recall@{k}: {test_metrics[k]['recall']:.4f} | NDCG@{k}: {test_metrics[k]['ndcg']:.4f}"
    logger.info(test_msg)
    logger.info("Training completed!")

if __name__ == "__main__":
    main()