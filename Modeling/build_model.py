# Modeling/build_model.py
# ============================================================
# Build & Train Implicit ALS Model
# Folder: Modeling/build_model.py
# ============================================================

from implicit.als import AlternatingLeastSquares
from Configuration.config import CFG


class ALSModelBuilder:
    """
    Builds and trains AlternatingLeastSquares (implicit library).

    Input to train():
        user_item_matrix (CSR, shape = [num_users, num_items])

    implicit ALS expects item-user matrix internally,
    so we transpose before fitting.
    """

    def __init__(self):
        self.model = AlternatingLeastSquares(
            factors=CFG.als.factors,
            regularization=CFG.als.regularization,
            iterations=CFG.als.iterations,
            use_gpu=CFG.als.use_gpu,
            random_state=CFG.als.random_state
        )

    def train(self, user_item_matrix):
        """
        Train ALS model on item-user matrix.
        """
        # implicit expects (items x users)
        item_user_matrix = user_item_matrix.T.tocsr()

        self.model.fit(item_user_matrix)
        return self.model

    def get_model(self):
        return self.model
