# main.py
# ============================================================
# Full Pipeline Runner (Implicit ALS Movie Recommender)
# Location: project root
# ============================================================

import os
import pandas as pd

from Resources.resource_manager import ResourceManager
from Dataset.dataset_loader import DatasetLoader
from Modeling.build_model import ALSModelBuilder
from Modeling.performance import Evaluator
from Inference.infer import InferenceEngine
from Configuration.config import CFG


def main():
    print("\n========== Movie Recommender Pipeline Started ==========\n")

    # ---------------------------------
    # STEP 1: Load Data
    # ---------------------------------
    rm = ResourceManager()
    loader = DatasetLoader(rm)

    ratings, movies, tags, links = loader.load_data()

    print("[1] Data Loaded")
    print("    ratings:", ratings.shape)
    print("    movies :", movies.shape)

    # ---------------------------------
    # STEP 2: Encode Users & Items
    # ---------------------------------
    ratings, user_to_id, item_to_id, id_to_user, id_to_item = loader.encode_users_items(ratings)
    num_users = len(user_to_id)
    num_items = len(item_to_id)

    print(f"\n[2] Encoding Done  -> users={num_users}, items={num_items}")

    # ---------------------------------
    # STEP 3: Per-user Train/Test Split
    # ---------------------------------
    train_ratings, test_ratings = loader.user_train_test_split(ratings)

    print("\n[3] Train/Test Split Done")
    print("    train:", train_ratings.shape)
    print("    test :", test_ratings.shape)

    # ---------------------------------
    # STEP 4: Build Sparse Train Matrix
    # ---------------------------------
    train_matrix = loader.build_sparse_matrix(train_ratings, num_users, num_items)
    print("\n[4] Sparse Matrix Built")
    print("    train_matrix shape:", train_matrix.shape)

    # ---------------------------------
    # STEP 5: Train ALS Model
    # ---------------------------------
    builder = ALSModelBuilder()
    model = builder.train(train_matrix)

    print("\n[5] ALS Model Trained Successfully")

    # ---------------------------------
    # STEP 6: Evaluate Recall@K
    # ---------------------------------
    evaluator = Evaluator()
    recall_scores = evaluator.evaluate(model, train_matrix, test_ratings)

    print("\n========== Recall@K Results ==========")
    for k, v in recall_scores.items():
        print(f"Recall@{k}: {v:.4f}")

    # ---------------------------------
    # STEP 7: Sample Recommendations
    # ---------------------------------
    infer_engine = InferenceEngine()
    sample_users = ratings[CFG.data.user_col].unique()[:5]

    print("\n========== Sample Recommendations ==========")

    all_recs = []  # for saving to CSV

    for uid in sample_users:
        recs = infer_engine.recommend_movies(
            model=model,
            user_id=uid,
            user_to_id=user_to_id,
            id_to_item=id_to_item,
            movies_df=movies,
            user_item_matrix=train_matrix,
            n=CFG.eval.n_recommendations
        )

        print(f"\nUser {uid}:")
        if not recs:
            print("  No recommendations found.")
        else:
            for r in recs:
                print("  -", r)
                all_recs.append({"userId": uid, "movie": r})

    # ---------------------------------
    # STEP 8: Save Recommendations CSV
    # ---------------------------------
    if all_recs:
        df_recs = pd.DataFrame(all_recs)
        rm.save_recommendations(df_recs, filename="recommendations.csv")

    # ---------------------------------
    # STEP 9: Save artifacts for infer.py
    # ---------------------------------
    infer_engine.save_artifacts(
        model=model,
        user_to_id=user_to_id,
        id_to_item=id_to_item,
        save_dir="output"
    )

    print("\n========== Pipeline Finished Successfully ==========\n")


if __name__ == "__main__":
    main()
