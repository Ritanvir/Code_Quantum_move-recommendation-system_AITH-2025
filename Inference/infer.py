# Inference/infer.py
# ============================================================
# INFERENCE-READY (can be run directly by evaluator)
# Behavior:
#   1) If artifacts exist in output/, load and recommend.
#   2) If artifacts missing, auto-train ONCE, save artifacts,
#      then recommend. (fallback safety)
#
# Run:
#   python Inference/infer.py --userId 10 --topN 5
# ============================================================

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Make sure project root is on path when run from Inference/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from Configuration.config import CFG
from implicit.als import AlternatingLeastSquares


# ------------------------------------------------------------
# Load artifacts (model + mappings) if they exist
# ------------------------------------------------------------
def artifacts_exist(save_dir="output"):
    return all(
        os.path.exists(os.path.join(save_dir, f))
        for f in ["als_model.pkl", "user_to_id.pkl", "id_to_item.pkl"]
    )


def load_artifacts(save_dir="output"):
    with open(os.path.join(save_dir, "als_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(save_dir, "user_to_id.pkl"), "rb") as f:
        user_to_id = pickle.load(f)
    with open(os.path.join(save_dir, "id_to_item.pkl"), "rb") as f:
        id_to_item = pickle.load(f)
    return model, user_to_id, id_to_item


def save_artifacts(model, user_to_id, id_to_item, save_dir="output"):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "als_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(save_dir, "user_to_id.pkl"), "wb") as f:
        pickle.dump(user_to_id, f)
    with open(os.path.join(save_dir, "id_to_item.pkl"), "wb") as f:
        pickle.dump(id_to_item, f)


# ------------------------------------------------------------
# Build mappings & sparse matrix (for training or inference)
# ------------------------------------------------------------
def build_mappings(ratings_df):
    user_to_id = {u: i for i, u in enumerate(ratings_df[CFG.data.user_col].unique())}
    item_to_id = {m: i for i, m in enumerate(ratings_df[CFG.data.item_col].unique())}
    id_to_item = {i: m for m, i in item_to_id.items()}
    return user_to_id, item_to_id, id_to_item


def build_user_item_matrix(ratings_df, user_to_id, item_to_id):
    df = ratings_df.copy()
    df["user_encoded"] = df[CFG.data.user_col].map(user_to_id)
    df["item_encoded"] = df[CFG.data.item_col].map(item_to_id)
    df = df.dropna(subset=["user_encoded", "item_encoded"])

    num_users = len(user_to_id)
    num_items = len(item_to_id)

    mat = csr_matrix(
        (
            df[CFG.data.rating_col].astype(np.float32),
            (df["user_encoded"].astype(int), df["item_encoded"].astype(int)),
        ),
        shape=(num_users, num_items)
    )
    return mat


# ------------------------------------------------------------
# Auto-train fallback (only if artifacts missing)
# ------------------------------------------------------------
def train_and_create_artifacts(ratings_df):
    print("[Info] Artifacts not found. Training ALS once for inference...")

    user_to_id, item_to_id, id_to_item = build_mappings(ratings_df)
    user_item_matrix = build_user_item_matrix(ratings_df, user_to_id, item_to_id)
    item_user_matrix = user_item_matrix.T.tocsr()

    model = AlternatingLeastSquares(
        factors=CFG.als.factors,
        regularization=CFG.als.regularization,
        iterations=CFG.als.iterations,
        use_gpu=CFG.als.use_gpu,
        random_state=CFG.als.random_state
    )
    model.fit(item_user_matrix)

    save_artifacts(model, user_to_id, id_to_item, save_dir="output")
    print("[Info] Training done. Artifacts saved to output/")

    return model, user_to_id, id_to_item, user_item_matrix


# ------------------------------------------------------------
# Recommend
# ------------------------------------------------------------
def recommend_movies(model, user_id, user_to_id, id_to_item, movies_df, user_item_matrix, n=5):
    internal_user = user_to_id.get(user_id)
    if internal_user is None:
        return []

    item_ids, scores = model.recommend(
        userid=internal_user,
        user_items=user_item_matrix[internal_user],
        N=n,
        filter_already_liked_items=True
    )

    movie_ids = [id_to_item[i] for i in item_ids]

    titles = (
        movies_df[movies_df[CFG.data.item_col].isin(movie_ids)]
        .set_index(CFG.data.item_col)
        .loc[movie_ids]["title"]
        .tolist()
    )
    return titles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--userId", type=int, required=True)
    parser.add_argument("--topN", type=int, default=5)
    args = parser.parse_args()

    # Load data
    ratings = pd.read_csv(CFG.data.ratings_path)
    movies  = pd.read_csv(CFG.data.movies_path)

    # Case 1: artifacts exist -> pure inference
    if artifacts_exist("output"):
        model, user_to_id, id_to_item = load_artifacts("output")
        item_to_id = {mid: enc for enc, mid in id_to_item.items()}
        user_item_matrix = build_user_item_matrix(ratings, user_to_id, item_to_id)

    # Case 2: artifacts missing -> fallback train once
    else:
        model, user_to_id, id_to_item, user_item_matrix = train_and_create_artifacts(ratings)

    # Recommend
    recs = recommend_movies(model, args.userId, user_to_id, id_to_item, movies, user_item_matrix, n=args.topN)

    print(f"\nTop-{args.topN} recommendations for user {args.userId}:")
    if not recs:
        print("  No recommendations found (unknown user).")
    else:
        for r in recs:
            print("  -", r)


if __name__ == "__main__":
    main()
