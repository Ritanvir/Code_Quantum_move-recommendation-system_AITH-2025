# Inference/regression_infer_cli.py
# ============================================================
# CLI Inference for Bonus Regression Model
# - Loads trained regression model + stats from output/
# - Predicts rating for (userId, movieId)
#
# BEFORE RUNNING:
#   python bonus_main.py   -> creates output/regression_model.pkl and regression_stats.pkl
#
# Run:
#   python Inference/regression_infer_cli.py --userId 10 --movieId 50
# ============================================================

import os
import sys
import pickle
import argparse
import pandas as pd

# add project root
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
    parser = argparse.ArgumentParser(description="Bonus Regression Rating Predictor CLI")
    parser.add_argument("--userId", type=int, required=True, help="MovieLens userId")
    parser.add_argument("--movieId", type=int, required=True, help="MovieLens movieId")
    args = parser.parse_args()

    # Load model + stats
    model, stats = load_regression_artifacts("output")

    # Load data for feature context
    rm = ResourceManager()
    loader = RegressionDatasetLoader(rm)

    ratings, movies = loader.load_data()
    df, X, y, _ = loader.build_features(ratings, movies)

    # Inference engine
    infer = RegressionInference(stats, movies_df=movies)

    pred_rating = infer.predict_user_movie_rating(
        user_id=args.userId,
        movie_id=args.movieId,
        model=model,
        global_df=df
    )

    if pred_rating is None:
        print(f"Movie {args.movieId} not found!")
        return

    print("\n========== Bonus Regression Inference ==========")
    print(f"UserId : {args.userId}")
    print(f"MovieId: {args.movieId}")
    print(f"Predicted Rating: {pred_rating:.4f}")


if __name__ == "__main__":
    main()
