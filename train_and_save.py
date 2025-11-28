# train_and_save.py
# ============================================================
# Run ONCE by you. Creates:
#   output/als_model.pkl
#   output/user_to_id.pkl
#   output/id_to_item.pkl
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

from Configuration.config import CFG


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


def main():
    ratings = pd.read_csv(CFG.data.ratings_path)

    user_to_id, item_to_id, id_to_item = build_mappings(ratings)
    user_item_matrix = build_user_item_matrix(ratings, user_to_id, item_to_id)
    item_user_matrix = user_item_matrix.T.tocsr()

    model = AlternatingLeastSquares(
        factors=CFG.als.factors,
        regularization=CFG.als.regularization,
        iterations=CFG.als.iterations,
        use_gpu=CFG.als.use_gpu,
        random_state=CFG.als.random_state
    )
    model.fit(item_user_matrix)

    os.makedirs("output", exist_ok=True)
    with open("output/als_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("output/user_to_id.pkl", "wb") as f:
        pickle.dump(user_to_id, f)
    with open("output/id_to_item.pkl", "wb") as f:
        pickle.dump(id_to_item, f)

    print("[Done] Artifacts saved in output/")


if __name__ == "__main__":
    main()
