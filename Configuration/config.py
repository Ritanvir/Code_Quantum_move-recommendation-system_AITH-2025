# Configuration/config.py
# ============================================================
# Central configuration for Movie Recommendation (Implicit ALS)
# FIXED: no circular imports + default_factory ok
# ============================================================

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    ratings_path: str = "Resources/ratings.csv"   # or ratings_clean.csv
    movies_path: str  = "Resources/movies.csv"
    tags_path: Optional[str] = "Resources/tags.csv"
    links_path: Optional[str] = "Resources/links.csv"

    user_col: str = "userId"
    item_col: str = "movieId"
    rating_col: str = "rating"
    timestamp_col: str = "timestamp"

    test_ratio: float = 0.2
    min_ratings_per_user: int = 2
    split_seed: int = 42


@dataclass
class ALSConfig:
    factors: int = 64
    regularization: float = 0.05
    iterations: int = 30

    use_gpu: bool = False
    random_state: int = 42
    confidence_alpha: float = 1.0


@dataclass
class EvalConfig:
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    n_recommendations: int = 5


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    als: ALSConfig = field(default_factory=ALSConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


CFG = Config()
