# Inference/regression_infer.py
# ============================================================
# Inference for regression rating prediction
# ============================================================

import pandas as pd
from Dataset.regression_loader import RegressionDatasetLoader


class RegressionInference:
    def __init__(self, stats, movies_df):
        self.user_mean = stats["user_mean"]
        self.movie_mean = stats["movie_mean"]
        self.movie_count = stats["movie_count"]
        self.genre_cols = stats["genre_cols"]
        self.movies = movies_df

    def predict_user_movie_rating(self, user_id, movie_id, model, global_df):
        movie_row = self.movies[self.movies["movieId"] == movie_id]
        if movie_row.empty:
            return None

        title = movie_row.iloc[0]["title"]
        genres = movie_row.iloc[0]["genres"]

        year = RegressionDatasetLoader.extract_year(title)
        if year is None:
            year = global_df["year"].median()

        gvec = pd.Series(0, index=self.genre_cols)
        for g in str(genres).split("|"):
            if g in gvec.index:
                gvec[g] = 1

        user_mean_rating = self.user_mean.get(user_id, global_df["rating"].mean())
        movie_mean_rating = self.movie_mean.get(movie_id, global_df["rating"].mean())
        movie_rating_count = self.movie_count.get(movie_id, 0)

        row = pd.DataFrame([{
            "year": year,
            "user_mean_rating": user_mean_rating,
            "movie_mean_rating": movie_mean_rating,
            "movie_rating_count": movie_rating_count,
            **gvec.to_dict()
        }])

        return float(model.predict(row)[0])
