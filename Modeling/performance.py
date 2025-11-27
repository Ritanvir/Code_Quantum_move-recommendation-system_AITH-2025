# Modeling/performance.py
# ============================================================
# Model Evaluation (Recall@K) for Implicit ALS
# Folder: Modeling/performance.py
# ============================================================

import numpy as np
from collections import defaultdict
from Configuration.config import CFG


class Evaluator:
    """
    Evaluates ALS model using Recall@K.
    Uses per-user held-out test interactions as ground truth.
    """

    # --------------------------------------------------------
    # Build Ground Truth from test set
    # --------------------------------------------------------
    def build_ground_truth(self, test_ratings):
        """
        Returns:
            ground_truth[user_encoded] = set(item_encoded)
        """
        gt = defaultdict(set)
        for _, row in test_ratings.iterrows():
            gt[row["user_encoded"]].add(row["item_encoded"])
        return gt

    # --------------------------------------------------------
    # Recall@K
    # --------------------------------------------------------
    def recall_at_k(self, model, user_matrix, ground_truth, users, k_list):
        """
        model: trained implicit ALS
        user_matrix: train sparse matrix (users x items)
        ground_truth: dict(user -> set(test_items))
        users: list/array of users to evaluate
        k_list: list of K values

        Returns dict {k: mean_recall}
        """
        recalls = {k: [] for k in k_list}
        max_k = max(k_list)

        for user in users:
            true_items = ground_truth.get(user)
            if not true_items:
                continue

            # Get top max_k recommendations (do NOT filter already-liked for eval)
            rec_items, rec_scores = model.recommend(
                userid=user,
                user_items=user_matrix[user],
                N=max_k,
                filter_already_liked_items=False
            )

            rec_items = list(rec_items)

            for k in k_list:
                top_k = set(rec_items[:k])
                hits = top_k.intersection(true_items)
                recalls[k].append(len(hits) / len(true_items))

        # mean recall per k
        return {k: float(np.mean(v)) if len(v) else 0.0 for k, v in recalls.items()}

    # --------------------------------------------------------
    # Full Evaluation Wrapper
    # --------------------------------------------------------
    def evaluate(self, model, train_matrix, test_ratings):
        """
        Convenience wrapper used in main.py
        """
        ground_truth = self.build_ground_truth(test_ratings)
        users = test_ratings["user_encoded"].unique()
        k_values = list(CFG.eval.k_values)

        scores = self.recall_at_k(
            model=model,
            user_matrix=train_matrix,
            ground_truth=ground_truth,
            users=users,
            k_list=k_values
        )

        return scores
