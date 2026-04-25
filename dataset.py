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
        rqvae_path: str = "rqvae_best_model.pth",
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

class ColdStartEvalDataset(TigerDataset): # 或者继承 AmazonDataset
    def __init__(
        self,
        cold_start_type: str = "user", # 可选: 'user', 'item', 'both'
        user_max_inter: int = 5,       # 历史交互数 <= 5 视为冷启动用户
        item_max_freq: int = 10,       # 在训练集中出现次数 <= 10 视为冷启动物品
        **kwargs
    ):
        # ⚠️ 冷启动测试通常仅用于线下评估阶段
        if kwargs.get("train_test_split", "test") == "train":
            raise ValueError("冷启动数据集只能用于 'valid' 或 'test' split！")
            
        # 1. 先调用父类初始化，完成全量样本的提取和 Embedding 预计算
        super().__init__(**kwargs)
        
        self.cold_start_type = cold_start_type.lower()
        self.user_max_inter = user_max_inter
        self.item_max_freq = item_max_freq
        
        # 2. 执行冷启动样本过滤
        self._filter_cold_start_samples()

    def _filter_cold_start_samples(self) -> None:
        """从生成的 samples 中过滤出符合冷启动定义的测试样本"""
        # 统计所有物品在【训练集】中出现的频次 (模拟真实的冷门尾部物品)
        # 注意：训练集使用的是 items[:-2]
        item_train_freq = np.zeros(self.num_items, dtype=int)
        for _, items, _ in self.sequences:
            for it in items[:-2]:
                item_train_freq[it - 1] += 1
                
        original_len = len(self.samples)
        filtered_samples = []
        
        for sample in self.samples:
            history = sample['history']
            target = sample['target']
            
            # 条件 A：该用户是否为冷启动用户？
            # 我们通过当前样本的 history 长度近似判断
            is_cold_user = (len(history) + 1) <= self.user_max_inter
            
            # 条件 B：该 Target 物品是否为冷门物品？
            is_cold_item = item_train_freq[target] <= self.item_max_freq
            
            # 路由分配
            if self.cold_start_type == "user" and is_cold_user:
                filtered_samples.append(sample)
            elif self.cold_start_type == "item" and is_cold_item:
                filtered_samples.append(sample)
            elif self.cold_start_type == "both" and (is_cold_user and is_cold_item):
                filtered_samples.append(sample)
                
        self.samples = filtered_samples
        print(f"❄️ 冷启动过滤完成 [{self.cold_start_type} 模式] | 样本数从 {original_len} 锐减至 -> {len(self.samples)}")

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

import torch
import numpy as np
from typing import Dict, List

def hmsr_collate_fn(batch: List[Dict], max_seq_len: int = 50):
    """
    HMSR Training Collate Function.
    Generates fully aligned item-level and semantic-level tensors with shifted targets.
    """
    B = len(batch)
    sem_id_dim = 3  # RQ-VAE codebooks
    
    # ================= 1. 初始化 Item-Level Tensors (长度 L) =================
    user_ids = torch.zeros(B, dtype=torch.long)
    input_item_ids = torch.zeros((B, max_seq_len), dtype=torch.long)
    timestamps = torch.zeros((B, max_seq_len), dtype=torch.long)
    target_item_ids = torch.zeros((B, max_seq_len), dtype=torch.long)
    item_seq_mask = torch.zeros((B, max_seq_len), dtype=torch.long)

    # ================= 2. 初始化 Semantic-Level Tensors (长度 L * 3) =================
    max_sem_len = max_seq_len * sem_id_dim
    history_sids = torch.zeros((B, max_sem_len), dtype=torch.long)
    target_sids = torch.zeros((B, max_sem_len), dtype=torch.long)
    token_type_ids = torch.zeros((B, max_sem_len), dtype=torch.long)
    semantic_seq_mask = torch.zeros((B, max_sem_len), dtype=torch.long)

    for i, sample in enumerate(batch):
        user_ids[i] = sample['user_id']

        his_items = sample['history']
        tgt_item = sample['target']
        his_ts = sample['timestamps']
        his_sid = sample['history_sid']  # np.array shape: (L, 3)
        tgt_sid = sample['target_sid']   # np.array shape: (3,)

        # 序列截断 (保留最新的 max_seq_len 个交互)
        if len(his_items) > max_seq_len:
            his_items = his_items[-max_seq_len:]
            his_ts = his_ts[-max_seq_len:]
            his_sid = his_sid[-max_seq_len:]

        L = len(his_items)

        # --- 填充 Item-Level (左侧填充) ---
        input_item_ids[i, max_seq_len - L:] = torch.tensor(his_items, dtype=torch.long)
        timestamps[i, max_seq_len - L:] = torch.tensor(his_ts, dtype=torch.long)
        item_seq_mask[i, max_seq_len - L:] = 1
        
        # 训练特供：生成 targets 序列 (整体向后移位 1 步)
        tgt_seq = his_items[1:] + [tgt_item]
        target_item_ids[i, max_seq_len - L:] = torch.tensor(tgt_seq, dtype=torch.long)

        # --- 填充 Semantic-Level (展平后左侧填充) ---
        his_sid_flat = torch.from_numpy(his_sid).view(-1)
        flat_L = len(his_sid_flat)
        start_idx = max_sem_len - flat_L

        history_sids[i, start_idx:] = his_sid_flat
        token_type_ids[i, start_idx:] = torch.arange(flat_L) % sem_id_dim
        semantic_seq_mask[i, start_idx:] = 1

        # 训练特供：生成 targets 语义序列 (his_sid 移位 1 个 Item，并接上 tgt_sid)
        tgt_sid_seq = np.concatenate([his_sid[1:], tgt_sid[np.newaxis, :]], axis=0)
        target_sids[i, start_idx:] = torch.from_numpy(tgt_sid_seq).view(-1)

    return {
        "user_id": user_ids,                    # [B]
        "input_item_ids": input_item_ids,       # [B, L]
        "timestamps": timestamps,               # [B, L]
        "item_seq_mask": item_seq_mask,         # [B, L]
        "target_item_ids": target_item_ids,     # [B, L]
        "history_sid": history_sids,            # [B, L * 3]
        "token_type_ids": token_type_ids,       # [B, L * 3]
        "semantic_seq_mask": semantic_seq_mask, # [B, L * 3]
        "target_sids": target_sids              # [B, L * 3]
    }

def hmsr_eval_collate_fn(batch: List[Dict], max_seq_len: int = 50):
    """
    HMSR Evaluation Collate Function.
    Provides full sequences for input, but single elements for targets.
    """
    B = len(batch)
    sem_id_dim = 3
    
    # ================= 1. 初始化 Input Tensors =================
    user_ids = torch.zeros(B, dtype=torch.long)
    input_item_ids = torch.zeros((B, max_seq_len), dtype=torch.long)
    timestamps = torch.zeros((B, max_seq_len), dtype=torch.long)
    item_seq_mask = torch.zeros((B, max_seq_len), dtype=torch.long)

    max_sem_len = max_seq_len * sem_id_dim
    history_sids = torch.zeros((B, max_sem_len), dtype=torch.long)
    token_type_ids = torch.zeros((B, max_sem_len), dtype=torch.long)
    semantic_seq_mask = torch.zeros((B, max_sem_len), dtype=torch.long)

    # ================= 2. 初始化 Target Tensors (单一目标) =================
    target_item_ids = torch.zeros(B, dtype=torch.long)
    target_sids = torch.zeros((B, sem_id_dim), dtype=torch.long)

    for i, sample in enumerate(batch):
        user_ids[i] = sample['user_id']
        target_item_ids[i] = sample['target']
        target_sids[i] = torch.from_numpy(sample['target_sid'])

        his_items = sample['history']
        his_ts = sample['timestamps']
        his_sid = sample['history_sid']

        if len(his_items) > max_seq_len:
            his_items = his_items[-max_seq_len:]
            his_ts = his_ts[-max_seq_len:]
            his_sid = his_sid[-max_seq_len:]

        L = len(his_items)

        # --- Input Items ---
        input_item_ids[i, max_seq_len - L:] = torch.tensor(his_items, dtype=torch.long)
        timestamps[i, max_seq_len - L:] = torch.tensor(his_ts, dtype=torch.long)
        item_seq_mask[i, max_seq_len - L:] = 1

        # --- Input Semantics ---
        his_sid_flat = torch.from_numpy(his_sid).view(-1)
        flat_L = len(his_sid_flat)
        start_idx = max_sem_len - flat_L

        history_sids[i, start_idx:] = his_sid_flat
        token_type_ids[i, start_idx:] = torch.arange(flat_L) % sem_id_dim
        semantic_seq_mask[i, start_idx:] = 1

    return {
        "user_id": user_ids,                    # [B]
        "input_item_ids": input_item_ids,       # [B, L]
        "timestamps": timestamps,               # [B, L]
        "item_seq_mask": item_seq_mask,         # [B, L]
        "history_sid": history_sids,            # [B, L * 3]
        "token_type_ids": token_type_ids,       # [B, L * 3]
        "semantic_seq_mask": semantic_seq_mask, # [B, L * 3]
        "target_item_ids": target_item_ids,     # [B]       <-- 区别点
        "target_sids": target_sids              # [B, 3]    <-- 区别点
    }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    print("🚀 === 开始测试 HMSR Collate Functions ===")
    
    # 1. 实例化数据集 (请确保路径与你服务器一致)
    # 为了测试方便，我们将 max_seq_len 设为 10，观察起来更直观
    TEST_MAX_LEN = 10
    
    try:
        # 这里以 Train 模式初始化
        test_dataset = TigerDataset(
            root="../genrec/dataset/amazon",
            rqvae_path="/opt/data/private/hmz/rec/grec/rqvae_best_model.pth", 
            split="beauty", 
            train_test_split="train", 
            max_seq_len=TEST_MAX_LEN, 
            min_seq_len=2
        )
        
        # ==========================================
        # 测试 1: Training Collate (包含 shifted 序列目标)
        # ==========================================
        print("\n" + "="*50)
        print("🧪 测试 1: 训练阶段 Collate (hmsr_collate_fn)")
        print("="*50)
        
        train_loader = DataLoader(
            test_dataset, 
            batch_size=2,   # 抽出 2 个样本对比 Padding 效果
            shuffle=True, 
            # 注意：通过 lambda 将 TEST_MAX_LEN 传给 collate_fn
            collate_fn=lambda b: hmsr_collate_fn(b, max_seq_len=TEST_MAX_LEN) 
        )
        
        train_batch = next(iter(train_loader))
        for key, tensor in train_batch.items():
            # 格式化打印，对齐输出结果
            print(f"字段: {key:<20} | 维度: {tensor.shape} | 类型: {tensor.dtype}")
            
        print("\n[样例验证] 第 1 个样本的 item_seq_mask (左侧补0):")
        print(train_batch["item_seq_mask"][0].tolist())
        
        print(f"\n[样例验证] 第 1 个样本的 token_type_ids (期望是 0,1,2 循环):")
        print(train_batch["token_type_ids"][0].tolist())


        # ==========================================
        # 测试 2: Evaluation Collate (单一预测目标)
        # ==========================================
        print("\n" + "="*50)
        print("🧪 测试 2: 验证阶段 Collate (hmsr_eval_collate_fn)")
        print("="*50)
        
        # 强制修改当前数据集的切分模式，并重新生成样本 (仅供快速测试)
        test_dataset.train_test_split = "valid"
        test_dataset._generate_samples()
        
        eval_loader = DataLoader(
            test_dataset, 
            batch_size=2, 
            shuffle=False, 
            collate_fn=lambda b: hmsr_eval_collate_fn(b, max_seq_len=TEST_MAX_LEN) 
        )
        
        eval_batch = next(iter(eval_loader))
        for key, tensor in eval_batch.items():
            print(f"字段: {key:<20} | 维度: {tensor.shape} | 类型: {tensor.dtype}")
            
        print("\n[差异验证] 注意观察 target_item_ids 已经从 [B, L] 变成了 [B] 标量")
        print(f"Target Item IDs: {eval_batch['target_item_ids'].tolist()}")
        print(f"Target Semantic IDs:\n{eval_batch['target_sids'].tolist()}")

    except Exception as e:
        print(f"\n❌ 测试失败，请检查路径或配置: {e}")