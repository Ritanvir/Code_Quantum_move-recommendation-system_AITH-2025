# Inference/infer.py
# ============================================================
# Inference Engine for Implicit ALS
# - Recommend movies for a given user
# Folder: Inference/infer.py
# ============================================================

import os
import pickle
import pandas as pd

from Configuration.config import CFG


class InferenceEngine:
    """
    Inference helper:
    - recommend_movies() used in main pipeline
    - load_artifacts() used if you want standalone inference
    """

    # --------------------------------------------------------
    # Recommend Movies for a raw userId
    # --------------------------------------------------------
    def recommend_movies(
        self,
        model,
        user_id,
        user_to_id,
        id_to_item,
        movies_df,
        user_item_matrix,
        n=None
    ):
        """
        Returns list of recommended movie titles for raw user_id.
        """
        if n is None:
            n = CFG.eval.n_recommendations

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

    # --------------------------------------------------------
    # Save Artifacts (optional utility)
    # --------------------------------------------------------
    def save_artifacts(
        self,
        model,
        user_to_id,
        id_to_item,
        save_dir="output"
    ):
        """
        Save model + mappings so that infer.py can run standalone later.
        """
        os.makedirs(save_dir, exist_ok=True)

        # save model
        with open(os.path.join(save_dir, "als_model.pkl"), "wb") as f:
            pickle.dump(model, f)

        # save mappings
        with open(os.path.join(save_dir, "user_to_id.pkl"), "wb") as f:
            pickle.dump(user_to_id, f)

        with open(os.path.join(save_dir, "id_to_item.pkl"), "wb") as f:
            pickle.dump(id_to_item, f)

        print(f"[Saved] artifacts in '{save_dir}/'")

    # --------------------------------------------------------
    # Load Artifacts (optional utility)
    # --------------------------------------------------------
    def load_artifacts(self, save_dir="output"):
        """
        Load model + mappings for standalone inference.
        """
        with open(os.path.join(save_dir, "als_model.pkl"), "rb") as f:
            model = pickle.load(f)

        with open(os.path.join(save_dir, "user_to_id.pkl"), "rb") as f:
            user_to_id = pickle.load(f)

        with open(os.path.join(save_dir, "id_to_item.pkl"), "rb") as f:
            id_to_item = pickle.load(f)

        return model, user_to_id, id_to_item


# ============================================================
# Standalone Demo Run
# ============================================================
if __name__ == "__main__":
    """
    Running infer.py directly expects:
      output/als_model.pkl
      output/user_to_id.pkl
      output/id_to_item.pkl
    And data:
      Resources/ratings_clean.csv (or ratings.csv)
      Resources/movies.csv
    """

    from Resources.resource_manager import ResourceManager
    from Dataset.dataset_loader import DatasetLoader

    rm = ResourceManager()
    loader = DatasetLoader(rm)
    infer_engine = InferenceEngine()

    # load data
    ratings, movies, tags, links = loader.load_data()

    # encode & build matrix
    ratings, user_to_id, item_to_id, id_to_user, id_to_item = loader.encode_users_items(ratings)
    num_users = len(user_to_id)
    num_items = len(item_to_id)

    user_item_matrix = loader.build_sparse_matrix(ratings, num_users, num_items)

    # load trained model + mappings
    model, user_to_id_saved, id_to_item_saved = infer_engine.load_artifacts("output")

    # take a sample user from dataset
    sample_user = ratings[CFG.data.user_col].unique()[0]

    recs = infer_engine.recommend_movies(
        model=model,
        user_id=sample_user,
        user_to_id=user_to_id_saved,
        id_to_item=id_to_item_saved,
        movies_df=movies,
        user_item_matrix=user_item_matrix,
        n=5
    )

    print(f"\nRecommendations for user {sample_user}:")
    for r in recs:
        print(" -", r)
