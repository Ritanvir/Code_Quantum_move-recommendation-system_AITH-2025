# Resources/resource_manager.py
# ============================================================
# Simple resource manager: load CSVs, save outputs
# ============================================================

import os
import pandas as pd


class ResourceManager:
    def __init__(self):
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

    def load_csv(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return pd.read_csv(path)

    def save_recommendations(self, df, filename="recommendations.csv"):
        out_path = os.path.join(self.output_dir, filename)
        df.to_csv(out_path, index=False)
        print(f"[Saved] recommendations -> {out_path}")
        return out_path
