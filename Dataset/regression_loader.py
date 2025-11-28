# Dataset/regression_loader.py
# ============================================================
# Loads ratings + movies, builds regression features
# ============================================================

import re
import pandas as pd
from Configuration.config import CFG


class RegressionDatasetLoader:
    def __init__(self, resource_manager):
        self.rm = resource_manager

    def load_data(self):
        ratings = self.rm.load_csv(CFG.data.ratings_path)
        movies  = self.rm.load_csv(CFG.data.movies_path)
        return ratings, movies

    @staticmethod
    def extract_year(title):
        m = re.search(r"\((\d{4})\)", str(title))
        return int(m.group(1)) if m else None

    def build_features(self, ratings, movies):
        # merge
        df = ratings.merge(movies, on="movieId", how="left")
        df["genres"] = df["genres"].fillna("")
        df["year"] = df["title"].apply(self.extract_year)

        # genre one-hot
        genre_dummies = df["genres"].str.get_dummies(sep="|")
        df = pd.concat([df, genre_dummies], axis=1)

        # user mean
        user_mean = df.groupby("userId")["rating"].mean()
        df["user_mean_rating"] = df["userId"].map(user_mean)

        # movie mean
        movie_mean = df.groupby("movieId")["rating"].mean()
        df["movie_mean_rating"] = df["movieId"].map(movie_mean)

        # movie count popularity
        movie_count = df.groupby("movieId")["rating"].count()
        df["movie_rating_count"] = df["movieId"].map(movie_count)

        # fill missing year
        df["year"] = df["year"].fillna(df["year"].median())

        feature_cols = (
            ["year", "user_mean_rating", "movie_mean_rating", "movie_rating_count"]
            + list(genre_dummies.columns)
        )

        X = df[feature_cols].copy()
        y = df["rating"].copy()

        # return everything needed for inference
        stats = {
            "user_mean": user_mean,
            "movie_mean": movie_mean,
            "movie_count": movie_count,
            "genre_cols": list(genre_dummies.columns)
        }

        return df, X, y, stats
