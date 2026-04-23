"""
Amazon Dataset for SASRec, HSTU, and TIGER (RQ-VAE) training.
Generates sequences, timestamps, semantic text embeddings, and discrete semantic IDs.
"""
import os
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from tqdm import tqdm
import logging

from amazon import DATASET_CONFIGS, parse_gzip_json, AMAZON_REVIEW_BASE_URL, download_file
from rqvae import RQVAE

logger = logging.getLogger(__name__)

class AmazonDataset(Dataset):
    def __init__(
        self,
        root: str = "dataset/amazon",
        split: str = "beauty",
        rqvae_weight_path: Optional[str] = "rqvae_best_model.pth",
        train_test_split: str = "train",
        max_seq_len: int = 50,
        min_seq_len: int = 5,
        use_text_embeddings: bool = True,
        encoder_model_name: str = "/opt/data/private/hmz/rec/genrec/models_hub/sentence-t5-xl",
        force_regenerate: bool = False,
    ) -> None:
        self.root = root
        self.split = split.lower()
        self.train_test_split = train_test_split
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        
        self.use_text_embeddings = use_text_embeddings
        self.encoder_model_name = encoder_model_name
        self.force_regenerate = force_regenerate

        # 1. 加载交互序列并建立 ID 映射 (Item 从 1 开始)
        self._load_sequences()
        
        # 2. 加载或生成文本语义 Embeddings (第 0 行全 0 作为 Padding)
        if self.use_text_embeddings:
            self._load_or_generate_embeddings()

            # 3. 🌟 高性能预计算：加载权重，一次性提取所有 Semantic IDs
            print(f"Loading RQ-VAE weights from {rqvae_weight_path}...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            hidden_dims_list = [512, 256, 128, 64, 32]
            self.rqvae = RQVAE(
                input_dim=768, 
                hidden_dims=hidden_dims_list, 
                num_codebooks=3, 
                codebook_size=256
            ).to(device)
            
            # 加载预训练权重
            if os.path.exists(rqvae_weight_path):
                self.rqvae.load_state_dict(torch.load(rqvae_weight_path, map_location=device, weights_only=False))
                print("✅ RQ-VAE 权重加载完成！")
            else:
                print("⚠️ 未找到 RQ-VAE 权重，将使用随机初始化提取 (仅供测试)")

            self.rqvae.eval()
            
            with torch.no_grad():
                # self.item_embeddings 形状为 [num_items + 1, 768]
                # 这样提取出来的 semantic_indices 形状也是 [num_items + 1, 3]
                # 刚好让 semantic_indices[item_id] 完美对齐，不需要 i-1！
                z = self.rqvae.encoder(self.item_embeddings.to(device))
                _, semantic_indices = self.rqvae.rq(z)  
                self.semantic_indices = torch.stack(semantic_indices, dim=1).cpu().numpy() 
                self.semantic_indices = self.semantic_indices
                print(self.semantic_indices[6845])
        # 4. 生成切片样本 (超快速的 Array Indexing)
        self._generate_samples()

    def _load_sequences(self) -> None:
        """Load user interaction sequences and build ID mappings."""
        config = DATASET_CONFIGS[self.split]
        self.raw_dir = os.path.join(self.root, "raw", self.split)
        os.makedirs(self.raw_dir, exist_ok=True)
        
        reviews_path = os.path.join(self.raw_dir, config["reviews"])
        
        if not os.path.exists(reviews_path):
            reviews_url = f"{AMAZON_REVIEW_BASE_URL}/{config['reviews']}"
            print(f"Downloading {reviews_url}...")
            download_file(reviews_url, reviews_path)

        user_sequences: Dict[int, List[tuple]] = {}
        self.item_id_mapping: Dict[str, int] = {}
        self.user_id_mapping: Dict[str, int] = {} 

        print(f"Loading sequences from {reviews_path} for {self.train_test_split}...")
        for review in tqdm(parse_gzip_json(reviews_path), desc="Processing reviews"):
            asin = review.get('asin')
            user_id = review.get('reviewerID')
            timestamp = review.get('unixReviewTime', 0)

            if asin and user_id:
                if asin not in self.item_id_mapping:
                    self.item_id_mapping[asin] = len(self.item_id_mapping) + 1
                if user_id not in self.user_id_mapping:
                    self.user_id_mapping[user_id] = len(self.user_id_mapping) + 1 

                item_id = self.item_id_mapping[asin]
                uid_int = self.user_id_mapping[user_id]

                if uid_int not in user_sequences:
                    user_sequences[uid_int] = []
                user_sequences[uid_int].append((timestamp, item_id))

        self.sequences = []
        for uid_int, seq in user_sequences.items():
            seq.sort(key=lambda x: x[0])
            timestamps = [x[0] for x in seq]
            items = [x[1] for x in seq]
            if len(items) >= self.min_seq_len:
                self.sequences.append((uid_int, items, timestamps))

        self.num_items = len(self.item_id_mapping)
        self.num_users = len(self.user_id_mapping)
        self.id2asin = {v: k for k, v in self.item_id_mapping.items()}
        print(f"Loaded {len(self.sequences)} sequences, {self.num_items} items, {self.num_users} users")

    def _generate_samples(self) -> None:
        """Generate sliding-window / LOO samples."""
        self.samples = []

        if self.train_test_split == "train":
            for uid, items, timestamps in tqdm(self.sequences, desc="Generating train samples"):
                items = [i-1 for i in items]  # 将 Item ID 从 1-based 转为 0-based
                items = items[:-2]
                
                timestamps = timestamps[:-2]
                if len(items) < 2:
                    continue
                self.samples.append({
                    'user_id': uid,
                    'history': items[:-1],
                    'timestamps': timestamps[:-1],
                    # 🌟 极速查表：不需要再到 getitem 里跑前向传播了！
                    'history_emb': self.item_embeddings[items[:-1]] if hasattr(self, 'item_embeddings') else None,
                    'history_sid': self.semantic_indices[items[:-1]] if hasattr(self, 'semantic_indices') else None,
                    'target': items[-1],
                    'target_sid': self.semantic_indices[items[-1]] if hasattr(self, 'semantic_indices') else None,
                })
        elif self.train_test_split == "valid":
            for uid, items, timestamps in self.sequences:
                items = [i-1 for i in items]  # 将 Item ID 从 1-based 转为 0-based
                items = items[:-1]
                
                timestamps = timestamps[:-1]
                if len(items) < 2:
                    continue
                start = max(0, len(items) - 1 - self.max_seq_len)
                self.samples.append({
                    'user_id': uid,
                    'history': items[start:-1],
                    'history_emb': self.item_embeddings[items[start:-1]] if hasattr(self, 'item_embeddings') else None,
                    'timestamps': timestamps[start:-1],
                    'history_sid': self.semantic_indices[items[start:-1]] if hasattr(self, 'semantic_indices') else None,
                    'target': items[-1],
                    'target_sid': self.semantic_indices[items[-1]] if hasattr(self, 'semantic_indices') else None,
                })
        else:  # test
            for uid, items, timestamps in self.sequences:
                if len(items) < 2:
                    continue
                items = [i-1 for i in items]    
                
                start = max(0, len(items) - 1 - self.max_seq_len)
                self.samples.append({
                    'user_id': uid,
                    'history': items[start:-1],
                    'history_emb': self.item_embeddings[items[start:-1]] if hasattr(self, 'item_embeddings') else None,
                    'timestamps': timestamps[start:-1],
                    'history_sid': self.semantic_indices[items[start:-1]] if hasattr(self, 'semantic_indices') else None,
                    'target': items[-1],
                    'target_sid': self.semantic_indices[items[-1]] if hasattr(self, 'semantic_indices') else None,
                })

        print(f"Generated {len(self.samples)} {self.train_test_split} samples")

    def _load_or_generate_embeddings(self) -> None:
        """Loads semantic item embeddings, regenerating if necessary."""
        self.processed_dir = os.path.join(self.root, "processed", self.split)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        encoder_short_name = os.path.basename(self.encoder_model_name.rstrip("/"))
        self.parquet_path = os.path.join(self.processed_dir, f"item_emb_{encoder_short_name}.parquet")
        
        if os.path.exists(self.parquet_path):
            print(f"Loading item embeddings from {self.parquet_path}")
            self.item_embeddings = pd.read_parquet(self.parquet_path)
            self.item_embeddings = torch.tensor(self.item_embeddings['embedding'].tolist(), dtype=torch.float32)
            print(self.item_embeddings.shape)
            return


        print("Generating new item embeddings from metadata...")
        from sentence_transformers import SentenceTransformer
        
        config = DATASET_CONFIGS[self.split]
        meta_path = os.path.join(self.raw_dir, config["meta"])
        
        if not os.path.exists(meta_path):
            meta_url = f"{AMAZON_REVIEW_BASE_URL}/{config['meta']}"
            download_file(meta_url, meta_path)

        item_info: Dict[str, dict] = {}
        for meta in tqdm(parse_gzip_json(meta_path), desc="Processing metadata"):
            asin = meta.get('asin')
            if asin in self.item_id_mapping:
                item_info[asin] = {
                    'title': meta.get('title', ''),
                    'price': meta.get('price', ''),
                    'salesRank': meta.get('salesRank', ''),
                    'brand': meta.get('brand', ''),
                    'categories': meta.get('categories', ''),
                }
        
        texts = []
        for item_id in range(1, self.num_items + 1):
            asin = self.id2asin[item_id]
            info = item_info.get(asin, {})
            semantics = (
                f"'title':{info.get('title', '')}\n"
                f" 'price':{info.get('price', '')}\n"
                f" 'brand':{info.get('brand', '')}\n"
                f" 'categories':{info.get('categories', '')}"
            )
            texts.append(semantics)

        model = SentenceTransformer(self.encoder_model_name)
        encoded_embs = model.encode(texts, batch_size=64, show_progress_bar=True)
        
        encoded_tensor = torch.tensor(encoded_embs, dtype=torch.float32)
        self.embed_dim = encoded_tensor.shape[-1]
        
        self.item_embeddings = torch.zeros((self.num_items + 1, self.embed_dim), dtype=torch.float32)
        self.item_embeddings[1:] = encoded_tensor
        
        save_list = []
        for item_id in range(0, self.num_items+1):
            save_list.append({
                'ItemID': item_id,
                'embedding': self.item_embeddings[item_id].numpy().tolist()
            })
        df = pd.DataFrame(save_list)
        df.to_parquet(self.parquet_path, index=False)
        print(f"Saved item embeddings to {self.parquet_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


class TigerDataset(Dataset):
    def __init__(
        self,
        root: str = "dataset/amazon",
        rqvae_path: str = None,
        split: str = "beauty",
        device: Optional[torch.device] = None,
        train_test_split: str = "train",
        max_seq_len: int = 50,
        min_seq_len: int = 5,
        use_text_embeddings: bool = True,
        encoder_model_name: str = "/opt/data/private/hmz/rec/genrec/models_hub/sentence-t5-xl",
        force_regenerate: bool = False,
        
    ) -> None:
        self.root = root
        self.split = split.lower()
        self.train_test_split = train_test_split
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        
        self.use_text_embeddings = use_text_embeddings
        self.encoder_model_name = encoder_model_name
        self.force_regenerate = force_regenerate

        # 1. 加载交互序列并建立 ID 映射 (Item 从 1 开始)
        self._load_sequences()
        
        # 2. 加载或生成文本语义 Embeddings (第 0 行全 0 作为 Padding)
        if self.use_text_embeddings:
            self._load_or_generate_embeddings()

            # 3. 🌟 高性能预计算：加载权重，一次性提取所有 Semantic IDs
            print(f"Loading RQ-VAE weights from {rqvae_path}...")
            
            
            hidden_dims_list = [512, 256, 128, 64, 32]
            self.rqvae = RQVAE(
                input_dim=768, 
                hidden_dims=hidden_dims_list, 
                num_codebooks=3, 
                codebook_size=256
            ).to(device)
            
            # 加载预训练权重
            if os.path.exists(rqvae_path):
                self.rqvae.load_state_dict(torch.load(rqvae_path, map_location=device))
                print("✅ RQ-VAE 权重加载完成！")
            else:
                print("⚠️ 未找到 RQ-VAE 权重，将使用随机初始化提取 (仅供测试)")

            self.rqvae.eval()
            
            with torch.no_grad():
                # self.item_embeddings 形状为 [num_items + 1, 768]
                # 这样提取出来的 semantic_indices 形状也是 [num_items + 1, 3]
                # 刚好让 semantic_indices[item_id] 完美对齐，不需要 i-1！
                z = self.rqvae.encoder(self.item_embeddings.to(device))
                _, semantic_indices = self.rqvae.rq(z)  
                self.semantic_indices = torch.stack(semantic_indices, dim=1).cpu().numpy() 
                self.semantic_indices = self.semantic_indices
                print(self.semantic_indices[6845])
        # 4. 生成切片样本 (超快速的 Array Indexing)
        self._generate_samples()

    def _load_sequences(self) -> None:
        """Load user interaction sequences and build ID mappings."""
        config = DATASET_CONFIGS[self.split]
        self.raw_dir = os.path.join(self.root, "raw", self.split)
        os.makedirs(self.raw_dir, exist_ok=True)
        
        reviews_path = os.path.join(self.raw_dir, config["reviews"])
        
        if not os.path.exists(reviews_path):
            reviews_url = f"{AMAZON_REVIEW_BASE_URL}/{config['reviews']}"
            print(f"Downloading {reviews_url}...")
            download_file(reviews_url, reviews_path)

        user_sequences: Dict[int, List[tuple]] = {}
        self.item_id_mapping: Dict[str, int] = {}
        self.user_id_mapping: Dict[str, int] = {} 

        print(f"Loading sequences from {reviews_path} for {self.train_test_split}...")
        for review in tqdm(parse_gzip_json(reviews_path), desc="Processing reviews"):
            asin = review.get('asin')
            user_id = review.get('reviewerID')
            timestamp = review.get('unixReviewTime', 0)

            if asin and user_id:
                if asin not in self.item_id_mapping:
                    self.item_id_mapping[asin] = len(self.item_id_mapping) + 1
                if user_id not in self.user_id_mapping:
                    self.user_id_mapping[user_id] = len(self.user_id_mapping) + 1 

                item_id = self.item_id_mapping[asin]
                uid_int = self.user_id_mapping[user_id]

                if uid_int not in user_sequences:
                    user_sequences[uid_int] = []
                user_sequences[uid_int].append((timestamp, item_id))

        self.sequences = []
        for uid_int, seq in user_sequences.items():
            seq.sort(key=lambda x: x[0])
            timestamps = [x[0] for x in seq]
            items = [x[1] for x in seq]
            if len(items) >= self.min_seq_len:
                self.sequences.append((uid_int, items, timestamps))

        self.num_items = len(self.item_id_mapping)
        self.num_users = len(self.user_id_mapping)
        self.id2asin = {v: k for k, v in self.item_id_mapping.items()}
        print(f"Loaded {len(self.sequences)} sequences, {self.num_items} items, {self.num_users} users")

    def _generate_samples(self) -> None:
        """Generate sliding-window / LOO samples."""
        self.samples = []

        if self.train_test_split == "train":
            for uid, items, timestamps in tqdm(self.sequences, desc="Generating train samples"):
                # Exclude last 2 items (for valid and test)
                train_items = [i-1 for i in items][:-2]  # 将 Item ID 从 1-based 转为 0-based 并排除最后两个
                train_timestamps = timestamps[:-2]
                if len(train_items) < 2:
                    continue
                # Generate sliding window samples
                for i in range(1, len(train_items)):
                    self.samples.append({
                        'user_id': uid,
                        'history': train_items[:i],
                        'timestamps': train_timestamps[:i],
                        # 🌟 极速查表：不需要再到 getitem 里跑前向传播了！
                        'history_emb': self.item_embeddings[train_items[:i]] if hasattr(self, 'item_embeddings') else None,
                        'history_sid': self.semantic_indices[train_items[:i]] if hasattr(self, 'semantic_indices') else None,
                        'target': train_items[i],
                        'target_sid': self.semantic_indices[train_items[i]] if hasattr(self, 'semantic_indices') else None,
                    })
        elif self.train_test_split == "valid":
            for uid, items, timestamps in self.sequences:
                items = [i-1 for i in items]  # 将 Item ID 从 1-based 转为 0-based
                items = items[:-1]
                
                timestamps = timestamps[:-1]
                if len(items) < 2:
                    continue
                start = max(0, len(items) - 1 - self.max_seq_len)
                self.samples.append({
                    'user_id': uid,
                    'history': items[start:-1],
                    'timestamps': timestamps[start:-1],
                    'history_emb': self.item_embeddings[items[start:-1]] if hasattr(self, 'item_embeddings') else None,
                    'history_sid': self.semantic_indices[items[start:-1]] if hasattr(self, 'semantic_indices') else None,
                    'target': items[-1],
                    'target_sid': self.semantic_indices[items[-1]] if hasattr(self, 'semantic_indices') else None,
                })
        else:  # test
            for uid, items, timestamps in self.sequences:
                if len(items) < 2:
                    continue
                items = [i-1 for i in items]    
                
                start = max(0, len(items) - 1 - self.max_seq_len)
                self.samples.append({
                    'user_id': uid,
                    'history': items[start:-1],
                    'timestamps': timestamps[start:-1],
                    'history_emb': self.item_embeddings[items[start:-1]] if hasattr(self, 'item_embeddings') else None,
                    'history_sid': self.semantic_indices[items[start:-1]] if hasattr(self, 'semantic_indices') else None,
                    'target': items[-1],
                    'target_sid': self.semantic_indices[items[-1]] if hasattr(self, 'semantic_indices') else None,
                })

        print(f"Generated {len(self.samples)} {self.train_test_split} samples")

    def _load_or_generate_embeddings(self) -> None:
        """Loads semantic item embeddings, regenerating if necessary."""
        self.processed_dir = os.path.join(self.root, "processed", self.split)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        encoder_short_name = os.path.basename(self.encoder_model_name.rstrip("/"))
        self.parquet_path = os.path.join(self.processed_dir, f"item_emb_{encoder_short_name}.parquet")
        
        if os.path.exists(self.parquet_path):
            print(f"Loading item embeddings from {self.parquet_path}")
            self.item_embeddings = pd.read_parquet(self.parquet_path)
            self.item_embeddings = torch.tensor(self.item_embeddings['embedding'].tolist(), dtype=torch.float32)
            print(self.item_embeddings.shape)
            return


        print("Generating new item embeddings from metadata...")
        from sentence_transformers import SentenceTransformer
        
        config = DATASET_CONFIGS[self.split]
        meta_path = os.path.join(self.raw_dir, config["meta"])
        
        if not os.path.exists(meta_path):
            meta_url = f"{AMAZON_REVIEW_BASE_URL}/{config['meta']}"
            download_file(meta_url, meta_path)

        item_info: Dict[str, dict] = {}
        for meta in tqdm(parse_gzip_json(meta_path), desc="Processing metadata"):
            asin = meta.get('asin')
            if asin in self.item_id_mapping:
                item_info[asin] = {
                    'title': meta.get('title', ''),
                    'price': meta.get('price', ''),
                    'salesRank': meta.get('salesRank', ''),
                    'brand': meta.get('brand', ''),
                    'categories': meta.get('categories', ''),
                }
        
        texts = []
        for item_id in range(1, self.num_items + 1):
            asin = self.id2asin[item_id]
            info = item_info.get(asin, {})
            semantics = (
                f"'title':{info.get('title', '')}\n"
                f" 'price':{info.get('price', '')}\n"
                f" 'brand':{info.get('brand', '')}\n"
                f" 'categories':{info.get('categories', '')}"
            )
            texts.append(semantics)

        model = SentenceTransformer(self.encoder_model_name)
        encoded_embs = model.encode(texts, batch_size=64, show_progress_bar=True)
        
        encoded_tensor = torch.tensor(encoded_embs, dtype=torch.float32)
        self.embed_dim = encoded_tensor.shape[-1]
        
        self.item_embeddings = torch.zeros((self.num_items + 1, self.embed_dim), dtype=torch.float32)
        self.item_embeddings[1:] = encoded_tensor
        
        save_list = []
        for item_id in range(0, self.num_items+1):
            save_list.append({
                'ItemID': item_id,
                'embedding': self.item_embeddings[item_id].numpy().tolist()
            })
        df = pd.DataFrame(save_list)
        df.to_parquet(self.parquet_path, index=False)
        print(f"Saved item embeddings to {self.parquet_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]



# ==========================================
# TIGER (Generative) Collate Function
# ==========================================
def tiger_collate_fn(batch):
    """
    将 TigerDataset 输出的单个样本打包成 Batch Tensor，并处理变长序列的 Padding。
    输入格式: {"history_semantic_ids": (L, 3), "target_semantic_ids": (3,)}
    """
    B = len(batch)
    sem_id_dim = 3  # RQ-VAE 生成的 codebook 数量
    
    # 找出当前 batch 中最长的历史序列长度 (注意这里是 item 的数量)
    max_item_len = max(len(x["history_sid"]) for x in batch)
    max_seq_len = max_item_len * sem_id_dim  # 展平后的真实 Token 长度
    
    # 初始化空的 Tensor (默认填充 pad_id = 0)
    item_input_ids = torch.zeros((B, max_seq_len), dtype=torch.long)
    token_type_ids = torch.zeros((B, max_seq_len), dtype=torch.long)
    seq_mask = torch.zeros((B, max_seq_len), dtype=torch.long)
    
    target_input_ids = torch.zeros((B, sem_id_dim), dtype=torch.long)
    # target_token_type 永远是固定的 [0, 1, 2]
    target_token_type_ids = torch.arange(sem_id_dim).unsqueeze(0).expand(B, -1).long()
    
    # 如果你的数据集没有 User ID，我们可以传入一个统一的 dummy ID (例如 0)
    user_input_ids = torch.tensor([[x["user_id"]] for x in batch])

    for i, sample in enumerate(batch):
        # 1. 将 (L, 3) 的二维数组展平成 (L*3,) 的一维序列
        his_flat = torch.from_numpy(sample["history_sid"]).view(-1)
        tgt_flat = torch.from_numpy(sample["target_sid"]).view(-1)
        
        L = len(his_flat)
        
        # 2. 采用【左侧填充】(Left Padding)，这对于生成式自回归模型更友好
        # 将真实序列放在最右边，左边留下的就是 0
        item_input_ids[i, max_seq_len - L:] = his_flat
        
        # 生成 token type: 0, 1, 2, 0, 1, 2 ...
        token_type_ids[i, max_seq_len - L:] = torch.arange(L) % sem_id_dim
        
        # 生成 mask: 真实数据处为 1，填充处为 0
        seq_mask[i, max_seq_len - L:] = 1
        
        # 目标直接赋值
        target_input_ids[i] = tgt_flat

    # 返回的字典 key 必须和 Tiger 模型的 forward 参数名严格对应
    return {
        "user_input_ids": user_input_ids,
        "item_input_ids": item_input_ids,
        "token_type_ids": token_type_ids,
        "target_input_ids": target_input_ids,
        "target_token_type_ids": target_token_type_ids,
        "seq_mask": seq_mask
    }        


def sasrec_collate_fn(batch: List[Dict], max_seq_len: int = 50, num_items: int = 0):
    """
    Collate function for SASRec.
    Pads sequences/timestamps to same length and creates input/target tensors.
    """
    histories = [b['history'] for b in batch]
    targets = [b['target'] for b in batch]
    timestamps_list = [b['timestamps'] for b in batch]

    max_len = min(max(len(h) for h in histories), max_seq_len)

    input_ids, target_ids = [], []
    input_ts, target_ts_list = [], []

    for history, target, ts in zip(histories, targets, timestamps_list):
        if len(history) > max_len:
            history = history[-max_len:]
            ts = ts[-max_len:]

        seq = history + [target]      
        ts_seq = ts + [ts[-1] if ts else 0]   

        pad_len = max_len + 1 - len(seq)
        padded_seq = [0] * pad_len + seq
        padded_ts = [0] * pad_len + ts_seq  

        input_ids.append(padded_seq[:-1])   
        target_ids.append(padded_seq[1:])    
        input_ts.append(padded_ts[:-1])
        target_ts_list.append(padded_ts[1:])

    result = {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'targets': torch.tensor(target_ids, dtype=torch.long),
        'input_ts': torch.tensor(input_ts, dtype=torch.long),
        'targets_ts': torch.tensor(target_ts_list, dtype=torch.long),
    }

    if num_items > 0:
        neg_ids = []
        for tgt_seq in target_ids:
            neg_seq = []
            for t in tgt_seq:
                if t == 0:  
                    neg_seq.append(0)
                else:
                    neg = random.randint(1, num_items)
                    while neg == t:
                        neg = random.randint(1, num_items)
                    neg_seq.append(neg)
            neg_ids.append(neg_seq)
        result['negatives'] = torch.tensor(neg_ids, dtype=torch.long)

    return result

def sasrec_eval_collate_fn(batch: List[Dict], max_seq_len: int = 50):
    """Collate function for SASRec evaluation."""
    histories = [b['history'] for b in batch]
    targets = [b['target'] for b in batch]
    timestamps_list = [b['timestamps'] for b in batch]

    max_len = min(max(len(h) for h in histories), max_seq_len)

    input_ids = []
    input_ts = []
    
    for history, ts in zip(histories, timestamps_list):
        if len(history) > max_len:
            history = history[-max_len:]
            ts = ts[-max_len:]
            
        pad_len = max_len - len(history)
        
        input_ids.append([0] * pad_len + history)
        input_ts.append([0] * pad_len + ts)

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'targets': torch.tensor(targets, dtype=torch.long),
        'input_ts': torch.tensor(input_ts, dtype=torch.long),
        # Note: SASRec validation usually doesn't strictly need target_ts, 
        # but to keep structure consistent, we omit it here just like HSTU.
    }


def hstu_collate_fn(batch: List[Dict], max_seq_len: int = 50):
    """
    Collate function for HSTU training.

    Pads sequences and timestamps, creates input/target tensors.
    """
    histories = [b['history'] for b in batch]
    timestamps_list = [b['timestamps'] for b in batch]
    targets = [b['target'] for b in batch]

    max_len = min(max(len(h) for h in histories), max_seq_len)

    input_ids = []
    target_ids = []
    timestamps = []

    for history, ts, target in zip(histories, timestamps_list, targets):
        if len(history) > max_len:
            history = history[-max_len:]
            ts = ts[-max_len:]

        seq = history + [target]
        ts_seq = ts + [ts[-1] if ts else 0]  # Use last timestamp for target

        pad_len = max_len + 1 - len(seq)
        padded_seq = [0] * pad_len + seq
        padded_ts = [0] * pad_len + ts_seq

        input_ids.append(padded_seq[:-1])
        target_ids.append(padded_seq[1:])
        timestamps.append(padded_ts[:-1])

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'targets': torch.tensor(target_ids, dtype=torch.long),
        'timestamps': torch.tensor(timestamps, dtype=torch.long),
    }


def hstu_eval_collate_fn(batch: List[Dict], max_seq_len: int = 50):
    """Collate function for HSTU evaluation."""
    histories = [b['history'] for b in batch]
    timestamps_list = [b['timestamps'] for b in batch]
    targets = [b['target'] for b in batch]

    max_len = min(max(len(h) for h in histories), max_seq_len)

    input_ids = []
    timestamps = []

    for history, ts in zip(histories, timestamps_list):
        if len(history) > max_len:
            history = history[-max_len:]
            ts = ts[-max_len:]

        pad_len = max_len - len(history)
        input_ids.append([0] * pad_len + history)
        timestamps.append([0] * pad_len + ts)

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'targets': torch.tensor(targets, dtype=torch.long),
        'timestamps': torch.tensor(timestamps, dtype=torch.long),
    }

def hmsr_collate_fn(batch: List[Dict], max_seq_len: int = 50, num_items: int = 0):
    """
    Collate function for HSTU.
    Similar to SASRec but also handles semantic IDs and token type IDs.
    """
    histories = [b['history'] for b in batch]
    targets = [b['target'] for b in batch]
    timestamps_list = [b['timestamps'] for b in batch]
    history_sids = [b['history_sid'] for b in batch]  
    target_sids = [b['target_sid'] for b in batch]      

    max_len = min(max(len(h) for h in histories), max_seq_len)

    item_input_ids, token_type_ids, seq_mask = [], [], []
    target_input_ids, target_token_type_ids = [], []

    for history, target, ts, his_sid, tgt_sid in zip(histories, targets, timestamps_list, history_sids, target_sids):
        if len(history) > max_len:
            history = history[-max_len:]
            ts = ts[-max_len:]
            his_sid = his_sid[-max_len:]

        pad_len = max_len - len(history)
        
        item_input_ids.append([0] * pad_len + history)
        token_type_ids.append([0] * pad_len + [i % 3 for i in range(len(history))])
        seq_mask.append([0] * pad_len + [1] * len(history))

        target_input_ids.append(tgt_sid)
        target_token_type_ids.append([0, 1, 2])  

    result = {
        'item_input_ids': torch.tensor(item_input_ids, dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        'seq_mask': torch.tensor(seq_mask, dtype=torch.long),
        'target_input_ids': torch.tensor(target_input_ids, dtype=torch.long),
        'target_token_type_ids': torch.tensor(target_token_type_ids, dtype=torch.long),
    }

    return result

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    # 记得把 rqvae_weight_path 换成你服务器上真实的路径
    dataset = TigerDataset(
        root="../genrec/dataset/amazon",
        rqvae_path="/opt/data/private/hmz/rec/grec/rqvae_best_model.pth",
        split="beauty", 
        train_test_split="train", 
        max_seq_len=50, 
        min_seq_len=5
    )
    print(dataset[0])
    print(len(dataset))

    dataset = AmazonDataset(
        root="../genrec/dataset/amazon",
        split="beauty", 
        train_test_split="train", 
        max_seq_len=50, 
        min_seq_len=5
    )
    print(dataset[0])