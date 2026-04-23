import os
# 禁用 Tokenizer 的底层并行，防止与 DataLoader 的多进程产生冲突
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from sasrec import SASRec
from dataset import AmazonDataset, sasrec_collate_fn, sasrec_eval_collate_fn
import warnings
warnings.filterwarnings("ignore")

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


def compute_metrics(logits, targets, k=10):
    """
    Compute Recall@K and NDCG@K.
    """
    # [FIX: Handle 3D logits from SASRec output (B, L, V)]
    if logits.dim() == 3:
        logits = logits[:, -1, :]  # (B, V) - prediction for last item
    
    batch_size = logits.size(0)
    
    # If targets is (batch_size, seq_len), take last item as ground truth
    if targets.dim() > 1:
        targets = targets[:, -1]  # (B,)
    
    # Get top-k predictions
    _, topk_indices = torch.topk(logits, k, dim=-1)  # (batch_size, k)
    
    # Convert targets to (batch_size, 1) for broadcasting
    target_tensor = targets.unsqueeze(1)  # (batch_size, 1)
    
    # Check if target is in top-k
    hit_matrix = (topk_indices == target_tensor).float()  # (batch_size, k)
    
    # Recall@K
    recall_at_k = hit_matrix.any(dim=1).float().mean().item()
    
    # NDCG@K
    discounts = 1.0 / torch.log2(torch.arange(2, k+2).float())  # (k,)
    dcg = (hit_matrix * discounts.to(hit_matrix.device)).sum(dim=1)  # (batch_size,)
    idcg = discounts[0]  # ideal DCG for single relevant item
    ndcg_at_k = (dcg / idcg).mean().item()
    
    return recall_at_k, ndcg_at_k


def evaluate(model, dataloader, device, num_items, k_list=None):
    """Evaluate model on validation/test set with ranking metrics for multiple K values."""
    if k_list is None:
        k_list = [10]
    
    model.eval()
    total_metrics = {k: {'recall': 0.0, 'ndcg': 0.0} for k in k_list}
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            
            output = model(input_ids)
            
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            if logits is not None:
                for k in k_list:
                    recall, ndcg = compute_metrics(logits, targets, k=k)
                    total_metrics[k]['recall'] += recall
                    total_metrics[k]['ndcg'] += ndcg
                num_batches += 1
    
    avg_loss = float('inf')  # 评估阶段不计算 loss
    avg_metrics = {}
    for k in k_list:
        avg_metrics[k] = {
            'recall': total_metrics[k]['recall'] / num_batches if num_batches > 0 else 0.0,
            'ndcg': total_metrics[k]['ndcg'] / num_batches if num_batches > 0 else 0.0
        }
    
    return avg_loss, avg_metrics


def train_epoch(model, dataloader, optimizer, scheduler, device, loss_type):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    num_samples = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        negatives = batch.get('negatives', None)
        if negatives is not None:
            negatives = negatives.to(device)
        
        _, loss = model(input_ids, targets, negatives)
        
        if loss is not None:
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item() * input_ids.size(0)
            num_samples += input_ids.size(0)
    
    avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train SASRec model')
    parser.add_argument('--dataset', type=str, default='beauty', help='Dataset name')
    parser.add_argument('--root', type=str, default='../genrec/dataset/amazon', help='Dataset root path')
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'bce'], help='Loss type')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--num_blocks', type=int, default=2, help='Number of transformer blocks')
    parser.add_argument('--ffn_dim', type=int, default=256, help='FFN hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--max_seq_len', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--eval_k', type=int, default=10, help='K for Recall@K and NDCG@K')
    args = parser.parse_args()
    # Recall@5: 0.0500, NDCG@5: 0.0318, Recall@10: 0.0740, NDCG@10: 0.0395
    # Recall@5: 0.0503, NDCG@5: 0.0362, Recall@10: 0.0717, NDCG@10: 0.0431
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.checkpoint_dir, 'train.log'))
    logger.info(f"Training arguments: {args}")
    
    logger.info("Loading datasets...")
    train_dataset = AmazonDataset(root=args.root, split=args.dataset, train_test_split="train", max_seq_len=args.max_seq_len)
    valid_dataset = AmazonDataset(root=args.root, split=args.dataset, train_test_split="valid", max_seq_len=args.max_seq_len)
    test_dataset = AmazonDataset(root=args.root, split=args.dataset, train_test_split="test", max_seq_len=args.max_seq_len)
    
    num_items = train_dataset.num_items
    logger.info(f"Number of items: {num_items}")
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Valid samples: {len(valid_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    collate_train = lambda x: sasrec_collate_fn(x, max_seq_len=args.max_seq_len, num_items=num_items if args.loss_type == "bce" else 0)
    collate_eval = lambda x: sasrec_eval_collate_fn(x, max_seq_len=args.max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_train, num_workers=4, pin_memory=True)
    test_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_eval, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_eval, num_workers=4, pin_memory=True)
    
    logger.info("Initializing model...")
    model = SASRec(
        num_items=num_items, max_seq_len=args.max_seq_len, embed_dim=args.embed_dim, 
        num_heads=args.num_heads, num_blocks=args.num_blocks, ffn_dim=args.ffn_dim, 
        dropout=args.dropout, loss_type=args.loss_type
    ).to(args.device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # [FIX 3: 改用 NDCG 作为 Early Stopping 和保存最佳模型的标准]
    best_valid_ndcg = -float('inf')
    patience_counter = 0
    
    logger.info("Starting training...")
    # 确保同时评估 K=5 和原始 eval_k
    eval_k_list = sorted(set([5, args.eval_k]))
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, None, args.device, args.loss_type)
        valid_loss, valid_metrics = evaluate(model, valid_loader, args.device, num_items, k_list=eval_k_list)
        
        # 获取原始 eval_k 的指标用于早停
        valid_recall_k = valid_metrics[args.eval_k]['recall']
        valid_ndcg_k = valid_metrics[args.eval_k]['ndcg']
        
        # 构建日志消息
        log_msg = f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}"
        for k in eval_k_list:
            log_msg += f", Recall@{k}: {valid_metrics[k]['recall']:.4f}, NDCG@{k}: {valid_metrics[k]['ndcg']:.4f}"
        logger.info(log_msg)
        
        # 使用原始 eval_k 的 NDCG 判断模型是否进步
        if valid_ndcg_k > best_valid_ndcg:
            best_valid_ndcg = valid_ndcg_k
            patience_counter = 0
            
            checkpoint_path = os.path.join(args.checkpoint_dir, f'sasrec_{args.dataset}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_ndcg': valid_ndcg_k,
                'args': args
            }, checkpoint_path)
            logger.info(f"Saved best model with NDCG@{args.eval_k}: {valid_ndcg_k:.4f} to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    logger.info("Loading best model for testing...")
    checkpoint_path = os.path.join(args.checkpoint_dir, f'sasrec_{args.dataset}_best.pth')
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_metrics = evaluate(model, test_loader, args.device, num_items, k_list=eval_k_list)
    test_msg = "Test Results"
    for k in eval_k_list:
        test_msg += f" - Recall@{k}: {test_metrics[k]['recall']:.4f}, NDCG@{k}: {test_metrics[k]['ndcg']:.4f}"
    logger.info(test_msg)
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
