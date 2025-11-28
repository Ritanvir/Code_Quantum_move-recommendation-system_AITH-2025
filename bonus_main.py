# bonus_main.py
# ============================================================
# BONUS ROUND Runner: Regression Rating Predictor
# ============================================================

import os
import pickle
from sklearn.model_selection import train_test_split

from Resources.resource_manager import ResourceManager
from Dataset.regression_loader import RegressionDatasetLoader
from Modeling.regression_model import RegressionModelBuilder
from Modeling.regression_performance import RegressionEvaluator
from Inference.regression_infer import RegressionInference
from Configuration.config import CFG


def main():
    print("\n========== BONUS ROUND: Regression Rating Predictor ==========\n")

    rm = ResourceManager()
    loader = RegressionDatasetLoader(rm)

    ratings, movies = loader.load_data()
    df, X, y, stats = loader.build_features(ratings, movies)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CFG.regression.test_size,
        random_state=CFG.regression.random_state
    )

    # model
    builder = RegressionModelBuilder()
    model = builder.build()
    model.fit(X_train, y_train)

    # predict + evaluate
    preds = model.predict(X_test)
    evaluator = RegressionEvaluator()
    scores = evaluator.evaluate(y_test, preds)

    print("===== Evaluation Metrics =====")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")

    # save model + stats
    os.makedirs("output", exist_ok=True)
    with open("output/regression_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("output/regression_stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    print("\n[Saved] regression_model.pkl + regression_stats.pkl -> output/")

    # demo inference
    infer = RegressionInference(stats, movies_df=movies)
    sample_user = df["userId"].unique()[0]
    sample_movie = df["movieId"].unique()[0]

    pred_rating = infer.predict_user_movie_rating(
        user_id=sample_user,
        movie_id=sample_movie,
        model=model,
        global_df=df
    )

    print(f"\nSample Prediction -> user {sample_user}, movie {sample_movie}")
    print("Predicted Rating:", pred_rating)


if __name__ == "__main__":
    main()
