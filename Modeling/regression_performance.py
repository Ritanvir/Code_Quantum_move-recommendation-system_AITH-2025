# Modeling/regression_performance.py
# ============================================================
# Evaluation: RMSE, MSE, MAE
# ============================================================

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


class RegressionEvaluator:
    def evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae
        }
