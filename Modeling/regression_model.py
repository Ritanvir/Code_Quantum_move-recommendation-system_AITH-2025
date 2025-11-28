# Modeling/regression_model.py
# ============================================================
# Regression models to predict user rating
# ============================================================

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from Configuration.config import CFG


class RegressionModelBuilder:
    def __init__(self):
        self.model_type = CFG.regression.model_type

    def build(self):
        if self.model_type == "ridge":
            model = Ridge(alpha=CFG.regression.ridge_alpha)

        elif self.model_type == "rf":
            model = RandomForestRegressor(
                n_estimators=CFG.regression.rf_estimators,
                max_depth=CFG.regression.rf_max_depth,
                random_state=CFG.regression.random_state,
                n_jobs=CFG.regression.rf_n_jobs
            )
        else:
            raise ValueError("model_type must be 'ridge' or 'rf'")

        return model
