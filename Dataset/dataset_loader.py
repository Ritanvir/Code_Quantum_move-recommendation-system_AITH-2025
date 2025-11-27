# Dataset/dataset_loader.py
# ============================================================
# Loads MovieLens data, encodes users/items,
# builds sparse matrices, and makes per-user train/test split
# Works with either:
#   Resources/ratings.csv  OR  Resources/ratings_clean.csv
# ============================================================

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from Configuration.config import CFG


class DatasetLoader:
    def __init__(self, resource_manager):
        self.rm = resource_manager

    # --------------------------------------------------------
    # Load CSV files
    # --------------------------------------------------------
    def load_data(self):
        ratings = self.rm.load_csv(CFG.data.ratings_path)
        movies  = self.rm.load_csv(CFG.data.movies_path)

        tags, links = None, None
        if getattr(CFG.data, "tags_path", None):
            try:
                tags = self.rm.load_csv(CFG.data.tags_path)
            except FileNotFoundError:
                pass

        if getattr(CFG.data, "links_path", None):
            try:
                links = self.rm.load_csv(CFG.data.links_path)
            except FileNotFoundError:
                pass

        return ratings, movies, tags, links

    # --------------------------------------------------------
    # Encode raw IDs -> internal contiguous IDs
    # --------------------------------------------------------
    def encode_users_items(self, ratings: pd.DataFrame):
        user_col = CFG.data.user_col
        item_col = CFG.data.item_col

        user_to_id = {u: i for i, u in enumerate(ratings[user_col].unique())}
        item_to_id = {m: i for i, m in enumerate(ratings[item_col].unique())}

        id_to_user = {i: u for u, i in user_to_id.items()}
        id_to_item = {i: m for m, i in item_to_id.items()}

        ratings["user_encoded"] = ratings[user_col].map(user_to_id)
        ratings["item_encoded"] = ratings[item_col].map(item_to_id)

        return ratings, user_to_id, item_to_id, id_to_user, id_to_item

    # --------------------------------------------------------
    # Build sparse matrix (users x items)
    # If "confidence" column exists, use that; else use rating
    # --------------------------------------------------------
    def build_sparse_matrix(self, ratings: pd.DataFrame, num_users: int, num_items: int):
        rating_col = "confidence" if "confidence" in ratings.columns else CFG.data.rating_col

        mat = csr_matrix(
            (
                ratings[rating_col].astype(np.float32),
                (ratings["user_encoded"], ratings["item_encoded"])
            ),
            shape=(num_users, num_items)
        )
        return mat

    # --------------------------------------------------------
    # Per-user train/test split
    # Users with < min_ratings_per_user stay fully in train
    # --------------------------------------------------------
    def user_train_test_split(self, ratings: pd.DataFrame):
        user_col = CFG.data.user_col
        test_ratio = CFG.data.test_ratio
        seed = CFG.data.split_seed
        min_ratings = CFG.data.min_ratings_per_user

        train_parts, test_parts = [], []

        for user, group in ratings.groupby(user_col):
            if len(group) < min_ratings:
                train_parts.append(group)
                continue

            train_g, test_g = train_test_split(
                group,
                test_size=test_ratio,
                random_state=seed
            )
            train_parts.append(train_g)
            test_parts.append(test_g)

        train_df = pd.concat(train_parts).reset_index(drop=True)
        test_df  = pd.concat(test_parts).reset_index(drop=True)

        return train_df, test_df
