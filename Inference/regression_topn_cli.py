# Inference/regression_topn_cli.py
# ============================================================
# Top-N Unseen Movie Recommender using Regression Predicted Rating
#
# BEFORE RUNNING:
#   python bonus_main.py
#   -> creates output/regression_model.pkl & regression_stats.pkl
#
# Run:
#   python Inference/regression_topn_cli.py --userId 10 --topN 10
# ============================================================

import os
import sys
import pickle
import argparse
import pandas as pd

# add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from Configuration.config import CFG
from Dataset.regression_loader import RegressionDatasetLoader
from Inference.regression_infer import RegressionInference
from Resources.resource_manager import ResourceManager


def load_regression_artifacts(save_dir="output"):
    model_path = os.path.join(save_dir, "regression_model.pkl")
    stats_path = os.path.join(save_dir, "regression_stats.pkl")

    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Artifacts missing in {save_dir}/.\n"
            f"Run: python bonus_main.py  first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    return model, stats


def main():
    parser = argparse.ArgumentParser(description="Top-N unseen movies via regression predicted rating")
    parser.add_argument("--userId", type=int, required=True, help="MovieLens userId")
    parser.add_argument("--topN", type=int, default=10, help="How many movies to recommend")
    args = parser.parse_args()

    # 1) Load artifacts
    model, stats = load_regression_artifacts("output")

    # 2) Load data
    rm = ResourceManager()
    loader = RegressionDatasetLoader(rm)
    ratings, movies = loader.load_data()
    df, X, y, _ = loader.build_features(ratings, movies)

    # 3) Prepare inference engine
    infer = RegressionInference(stats, movies_df=movies)

    # 4) Find movies user already rated
    rated_movie_ids = set(df[df["userId"] == args.userId]["movieId"].unique())

    # 5) Candidate unseen movies
    all_movie_ids = movies["movieId"].unique()
    unseen_movie_ids = [mid for mid in all_movie_ids if mid not in rated_movie_ids]

    if len(unseen_movie_ids) == 0:
        print("User already rated all movies!")
        return

    # 6) Predict rating for each unseen movie (can be slow, but OK for latest-small)
    predictions = []
    for mid in unseen_movie_ids:
        pred = infer.predict_user_movie_rating(
            user_id=args.userId,
            movie_id=mid,
            model=model,
            global_df=df
        )
        if pred is not None:
            predictions.append((mid, pred))

    # 7) Sort by predicted rating DESC
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_preds = predictions[:args.topN]

    # 8) Print results
    print("\n========== Top-N Unseen Movies (Regression) ==========")
    print(f"UserId: {args.userId}")
    print(f"TopN  : {args.topN}\n")

    for rank, (mid, score) in enumerate(top_preds, start=1):
        title = movies.loc[movies["movieId"] == mid, "title"].values[0]
        print(f"{rank:02d}. {title}  | Predicted Rating: {score:.4f}")


if __name__ == "__main__":
    main()
