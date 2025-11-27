# DataPreprocessing/preprocess.py
# ============================================================
# Preprocess MovieLens dataset for Implicit ALS
# Outputs: Resources/ratings_clean.csv
# ============================================================

import os
import pandas as pd
import numpy as np

from Configuration.config import CFG
from Resources.resource_manager import ResourceManager


class Preprocessor:
    def __init__(self):
        self.rm = ResourceManager()

    def load_raw(self):
        ratings = self.rm.load_csv(CFG.data.ratings_path)
        movies  = self.rm.load_csv(CFG.data.movies_path)

        tags, links = None, None
        if CFG.data.tags_path:
            try:
                tags = self.rm.load_csv(CFG.data.tags_path)
            except FileNotFoundError:
                pass

        if CFG.data.links_path:
            try:
                links = self.rm.load_csv(CFG.data.links_path)
            except FileNotFoundError:
                pass

        return ratings, movies, tags, links

    def clean_ratings(self, ratings: pd.DataFrame):
        # drop nulls
        ratings = ratings.dropna(subset=[
            CFG.data.user_col, CFG.data.item_col, CFG.data.rating_col
        ])

        # remove duplicates (keep latest if timestamp exists)
        if CFG.data.timestamp_col in ratings.columns:
            ratings = ratings.sort_values(CFG.data.timestamp_col)
        ratings = ratings.drop_duplicates(
            subset=[CFG.data.user_col, CFG.data.item_col],
            keep="last"
        )

        # ensure correct dtypes
        ratings[CFG.data.user_col] = ratings[CFG.data.user_col].astype(int)
        ratings[CFG.data.item_col] = ratings[CFG.data.item_col].astype(int)
        ratings[CFG.data.rating_col] = ratings[CFG.data.rating_col].astype(float)

        return ratings

    def filter_users_items(
        self,
        ratings: pd.DataFrame,
        min_user_ratings: int = 2,
        min_item_ratings: int = 2
    ):
        """
        Optional: remove very inactive users/items.
        Set mins higher (e.g., 5 or 10) if you want stronger signals.
        """
        user_counts = ratings.groupby(CFG.data.user_col).size()
        item_counts = ratings.groupby(CFG.data.item_col).size()

        keep_users = user_counts[user_counts >= min_user_ratings].index
        keep_items = item_counts[item_counts >= min_item_ratings].index

        filtered = ratings[
            ratings[CFG.data.user_col].isin(keep_users)
            & ratings[CFG.data.item_col].isin(keep_items)
        ].copy()

        return filtered

    def add_confidence(self, ratings: pd.DataFrame):
        """
        Implicit ALS expects confidence values.
        We use ratings * alpha as confidence.
        """
        alpha = CFG.als.confidence_alpha
        ratings["confidence"] = ratings[CFG.data.rating_col] * alpha
        return ratings

    def run(
        self,
        min_user_ratings: int = 2,
        min_item_ratings: int = 2,
        save_name: str = "ratings_clean.csv"
    ):
        ratings, movies, tags, links = self.load_raw()

        print("[Raw] ratings:", ratings.shape)
        ratings = self.clean_ratings(ratings)
        print("[Cleaned] ratings:", ratings.shape)

        ratings = self.filter_users_items(
            ratings,
            min_user_ratings=min_user_ratings,
            min_item_ratings=min_item_ratings
        )
        print("[Filtered] ratings:", ratings.shape)

        ratings = self.add_confidence(ratings)

        out_path = os.path.join("Resources", save_name)
        ratings.to_csv(out_path, index=False)
        print(f"[Saved] cleaned ratings -> {out_path}")

        return ratings


if __name__ == "__main__":
    pre = Preprocessor()
    pre.run(
        min_user_ratings=2,
        min_item_ratings=2,
        save_name="ratings_clean.csv"
    )
