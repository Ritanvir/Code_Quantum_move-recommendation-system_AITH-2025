
# Code_Quantum — Movie Recommendation System (Implicit ALS) + Bonus Regression

This repository contains two parts:

1) **Main Task:** Movie Recommendation System using **Implicit ALS** (Collaborative Filtering) on MovieLens.  
2) **Bonus Round:** Automated regression system to **predict user ratings** for unseen movies using movie attributes + historical ratings.

Both pipelines are modular and follow a cookiecutter-like structure.

---

##  Part 1: Main Task — Implicit ALS Recommender

### What it does
- Trains an **Implicit ALS** model on historical user ratings.
- Evaluates model using **Recall@K**.
- Generates Top-N movie recommendations for users.
- **Inference-ready:** Evaluator can run inference directly via `Inference/infer.py` without running training first **as long as artifacts exist**.

### Key files
- `train_and_save.py` → run once to generate model artifacts
- `Inference/infer.py` → inference-ready script
- `main.py` → full training + evaluation + sample inference pipeline
- `Modeling/build_model.py` → ALS model training logic
- `Modeling/performance.py` → Recall@K evaluation

---

##  Part 2: Bonus Round — Regression Rating Predictor

### What it does
- Trains regression model using historical user reviews.
- Predicts how a user might rate an **unseen movie** based on:
  - genres (one-hot)
  - release year
  - user average rating
  - movie average rating
  - movie popularity (rating count)
- Evaluates model using:
  - **RMSE**
  - **MSE**
  - **MAE** (extra metric)
- **Inference-ready CLI** for:
  - single user+movie rating prediction
  - Top-N unseen movies for a user ranked by predicted rating

### Key files
- `bonus_main.py` → trains regression model + saves artifacts
- `Inference/regression_infer_cli.py` → predict rating for (userId, movieId)
- `Inference/regression_topn_cli.py` → Top-N unseen movies for a user
- `Dataset/regression_loader.py` → feature builder
- `Modeling/regression_model.py` → regression model builder
- `Modeling/regression_performance.py` → RMSE/MSE/MAE evaluation

---

## Project Structure

```

Code_Quantum/
│
├── Configuration/
│   ├── **init**.py
│   └── config.py
│
├── DataPreprocessing/
│   ├── **init**.py
│   └── preprocess.py
│
├── Dataset/
│   ├── **init**.py
│   ├── dataset_loader.py
│   └── regression_loader.py
│
├── Modeling/
│   ├── **init**.py
│   ├── build_model.py
│   ├── performance.py
│   ├── regression_model.py
│   └── regression_performance.py
│
├── Inference/
│   ├── **init**.py
│   ├── infer.py
│   ├── regression_infer.py
│   ├── regression_infer_cli.py
│   └── regression_topn_cli.py
│
├── Resources/
│   ├── **init**.py
│   ├── resource_manager.py
│   ├── ratings.csv
│   ├── movies.csv
│   ├── tags.csv
│   └── links.csv
│
├── output/
│   ├── als_model.pkl
│   ├── user_to_id.pkl
│   ├── id_to_item.pkl
│   ├── recall_scores.csv
│   ├── recommendations.csv
│   ├── regression_model.pkl
│   └── regression_stats.pkl
│
├── main.py
├── train_and_save.py
├── bonus_main.py
├── requirements.txt
└── README.md

```

---

## Dataset

Using **MovieLens latest-small** dataset:

- `ratings.csv` → user ratings (historical reviews)
- `movies.csv`  → title + genres
- `tags.csv`    → optional metadata
- `links.csv`   → optional IMDb/TMDb ids

Place files here:

```

Resources/

````

---

## Setup

> **Important:** `implicit` works smoothly on Windows with Python **3.11/3.12**.

### 1) Create & activate venv
```powershell
python -m venv venv
.\venv\Scripts\activate
````

### 2) Install dependencies

```powershell
python -m pip install -U pip
python -m pip install pandas numpy scipy scikit-learn implicit matplotlib
```

Or:

```powershell
pip install -r requirements.txt
```

---

#  MAIN TASK USAGE (ALS)

## A) Train once & save artifacts (YOU run this)

```powershell
python train_and_save.py
```

This creates:

```
output/
  als_model.pkl
  user_to_id.pkl
  id_to_item.pkl
```

> Make sure these are pushed to GitHub (don’t ignore them).

---

## B) Inference directly (EVALUATOR runs this)

```powershell
python Inference/infer.py --userId 10 --topN 10
```

Output example:

```
Top-10 recommendations for user 10:
  - Toy Story (1995)
  - Heat (1995)
  ...
```

---

## C) Full training + evaluation pipeline (optional check)

```powershell
python main.py
```

Shows:

* train/test sizes
* Recall@K
* sample recommendations

Also saves:

```
output/recall_scores.csv
output/recommendations.csv
```

---

#  BONUS ROUND USAGE (Regression)

## A) Train regression model & save artifacts (YOU run this)

```powershell
python bonus_main.py
```

Creates:

```
output/regression_model.pkl
output/regression_stats.pkl
```

Prints:

* RMSE
* MSE
* MAE
* sample prediction

---

## B) Predict rating for a specific user-movie (CLI inference)

```powershell
python Inference/regression_infer_cli.py --userId 10 --movieId 50
```

Output:

```
Predicted Rating: 3.8421
```

---

## C) Top-N unseen movies for a user (ranked by predicted rating)

```powershell
python Inference/regression_topn_cli.py --userId 10 --topN 10
```

Output:

```
01. Shawshank Redemption (1994) | Predicted Rating: 4.61
02. ...
```

---

## Evaluation Metrics (Bonus)

* **MSE**
* **RMSE**
* **MAE** (extra chosen metric)

---

## Troubleshooting

###  implicit install error on Python 3.13

Fix: Use Python 3.11/3.12 and recreate venv.

###  Index out of bounds during ALS inference

Artifacts mismatch.
Fix:

1. delete `output/*.pkl`
2. run `python train_and_save.py` again
3. rerun inference.

###  Circular import error

Do not import project modules inside `config.py`.

---

## Author

**Razwanul Islam Tanvir**

