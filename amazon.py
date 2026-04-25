"""
Amazon Reviews Dataset for RQVAE training.
Supports automatic download and processing of Amazon Review 2014 5-core data.
"""
import gzip
import json
import logging
import os
import urllib.request
import numpy as np
import pandas as pd
import torch


from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Amazon Review 2014 download URLs
AMAZON_REVIEW_BASE_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"

DATASET_CONFIGS = {
    # Amazon 2014 datasets
    "beauty": {
        "reviews": "reviews_Beauty_5.json.gz",
        "meta": "meta_Beauty.json.gz",
    },
    "sports": {
        "reviews": "reviews_Sports_and_Outdoors_5.json.gz",
        "meta": "meta_Sports_and_Outdoors.json.gz",
    },
    "toys": {
        "reviews": "reviews_Toys_and_Games_5.json.gz",
        "meta": "meta_Toys_and_Games.json.gz",
    },
    "clothing": {
        "reviews": "reviews_Clothing_Shoes_and_Jewelry_5.json.gz",
        "meta": "meta_Clothing_Shoes_and_Jewelry.json.gz",
    },
    "home": {
        "reviews": "reviews_Home_and_Kitchen_5.json.gz",
        "meta": "meta_Home_and_Kitchen.json.gz",
    },
    # Amazon 2023 datasets (processed by genrec.data.amazon2023)
    "books": {
        "reviews": "reviews_Books_5.json.gz",
        "meta": "meta_Books.json.gz",
    },
    "arts_crafts_and_sewing": {
        "reviews": "reviews_Arts_Crafts_and_Sewing_5.json.gz",
        "meta": "meta_Arts_Crafts_and_Sewing.json.gz",
    },
}


def download_file(url: str, dest_path: str) -> None:
    """Download file with progress bar."""
    if os.path.exists(dest_path):
        logger.info(f"File already exists: {dest_path}")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logger.info(f"Downloading {url} -> {dest_path}")

    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get('content-length', 0))
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))

    logger.info(f"Downloaded: {dest_path}")


def parse_gzip_json(path: str):
    """Parse gzipped JSON file line by line."""
    with gzip.open(path, 'rt', encoding='utf-8') as g:
        for line in g:
            try:
                yield json.loads(line.strip())
            except json.JSONDecodeError:
                # Handle malformed lines
                try:
                    yield eval(line.strip())
                except:
                    continue







