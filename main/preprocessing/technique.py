import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import warnings
warnings.filterwarnings("ignore")

class HandDataSet:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path).dropna().reset_index(drop=True)

        self.angle_cols = [c for c in self.df.columns if c.startswith("angle_")]
        self.dist_cols  = [c for c in self.df.columns if c.startswith("dist_")]

        self.thumb_cols = [
            "thumb_y_rel_wrist",
            "thumb_y_rel_index_mcp",
            "thumb_dist_to_index_mcp",
            "thumb_palm_dist",
            "thumb_adduction_angle",
        ]

        self.base_features = self.angle_cols + self.dist_cols + self.thumb_cols
    
    def get_features(self, subset: list[str] | None = None):
        return self.df[subset if subset else self.base_features]

    def get_target(self, column: str):
        return self.df[column]

    def filter_errors(self, error_col: str):
        return self.df[self.df[error_col] == 1]

class TechniqueModel:
    def __init__(
        self,
        name: str,
        n_estimators: int = 300,
        class_weight: str = "balanced",
    ):
        self.name = name
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        )

    def train(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            stratify=y,
            test_size=test_size,
            random_state=42
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        print(f"\n📊 {self.name}")
        print(classification_report(y_test, y_pred))

        return X_test, y_test, y_pred


    def predict(self, X):
        return self.model.predict(X)

class TechniqueOrchestrator:
    def __init__(self, csv_path: str):
        self.dataset = HandDataSet(csv_path)

    def train_detector(self, part: str):
        model = TechniqueModel(f"{part.upper()} ERROR DETECTOR")

        X = self.dataset.get_features()
        y = self.dataset.get_target(f"{part}_error")

        model.train(X, y)
        return model

    def train_type_classifier(self, part: str):
        error_df = self.dataset.filter_errors(f"{part}_error")

        X = error_df[self.dataset.base_features]
        y = error_df[f"{part}_error_type"]

        model = TechniqueModel(f"{part.upper()} ERROR TYPE", n_estimators=200)
        model.train(X, y)
        return model
