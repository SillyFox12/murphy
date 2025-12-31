import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings("ignore")

class HandDataSet:
    def __init__(self, good_tech: list[str], bad_tech: list[str], part: str):
        dfs = []

        # Load GOOD technique
        for csv in good_tech:
            df = pd.read_csv(csv)
            df[f"{part}_error"] = 0
            df[f"{part}_error_type"] = "none"
            dfs.append(df)

        # Load BAD technique
        for csv in bad_tech:
            df = pd.read_csv(csv)
            df[f"{part}_error"] = 1
            dfs.append(df)

        self.df = (
            pd.concat(dfs, ignore_index=True)
              .dropna()
              .reset_index(drop=True)
        )

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
    
    def get_error_type_dataset(self, part: str):
        """
        Returns features + type labels for error-type classification.
        Only error rows are included.
        """
        error_df = self.df[self.df[f"{part}_error"] == 1]

        X = error_df[self.base_features]
        y = error_df[f"{part}_error_type"]

        return X, y


class TechniqueModel:
    def __init__(
        self,
        name: str,
        n_estimators: int = 200, 
        max_features: str = "log2", # Changed from 'sqrt' to 'log2' to de-correlate trees
        class_weight: str = "balanced_subsample", # More robust weighting
    ):
        self.name = name
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            class_weight=class_weight,
            
            # --- Regularization to prevent Overfitting ---
            max_depth=12,              # Prevent infinite depth (memorization)
            min_samples_split=10,      # Require 10 samples to make a new decision node
            min_samples_leaf=5,        # Leaves must have >5 samples (smooths boundaries)
            max_samples=0.8,           # Each tree only sees 80% of data (bootstrapping)
            
            random_state=42,
            n_jobs=-1,
            oob_score=True
        )

    def train(self, X, y, test_size=0.25):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            stratify=y,
            test_size=test_size,
            random_state=42
        )

        self.model.fit(X_train, y_train)
        
        return X_test, y_test

    def predict(self, X):
        return self.model.predict(X)

class TechniqueOrchestrator:
    def __init__(self, good_tech: list[str] | None = None, bad_tech: list[str] | None = None, part: str = None):
        self.part = part
        self.dataset = HandDataSet(good_tech=good_tech,
                                   bad_tech=bad_tech,
                                   part=part)

    def train_detector(self, part: str):    
        model = TechniqueModel(f"{part.upper()} ERROR DETECTOR")

        X = self.dataset.get_features()
        y = self.dataset.get_target(f"{part}_error")

        # Capture the test set specifically for evaluation
        X_test, y_test = model.train(X, y)
        
        # Predict only on the unseen test set for the report
        y_pred = model.predict(X_test)

        print(f"\n📊 {model.name} (Validation on Unseen Data)")
        print(classification_report(y_test, y_pred))
        
        # Return test data for the confusion matrix
        return model, X_test, y_test
    
    def train_type_classifier(self, part: str):
        model = TechniqueModel(
            name=f"{part.upper()} ERROR TYPE CLASSIFIER",
            n_estimators=300,
            class_weight="balanced"
        )

        X, y = self.dataset.get_error_type_dataset(part)

        # Sanity check
        print("\n🧪 Error type distribution:")
        print(y.value_counts())

        X_test, y_test = model.train(X, y)

        y_pred = model.predict(X_test)

        print(f"\n📊 {model.name} (Validation on Unseen Error Samples)")
        print(classification_report(y_test, y_pred))

        return model, X_test, y_test


    def save_model(self, model: TechniqueModel, filepath: str):
        joblib.dump(model, filepath)

    def plot_feature_importance(self, model: TechniqueModel, top_n: int = 10):
        importances = model.model.feature_importances_
        feature_names = self.dataset.base_features

        feat_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(top_n)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
        plt.title(f'Top {top_n} Feature Importances for {model.name}')
        plt.tight_layout()
        plt.show()
    
    def confusion_matrix(self, y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix (Test Set Only)')
        plt.show()

if __name__ == '__main__':
    good_tech_files = []
    bad_tech_files = ["data/processed_hand_data/wrist/raised/raised_wrist.csv", 
                      "data/processed_hand_data/wrist/caved/caved_wrist.csv"]
    
    orchestrator = TechniqueOrchestrator(good_tech=good_tech_files,
                                         bad_tech=bad_tech_files,
                                         part="wrist")  

    wrist_detector, X_test, y_test = orchestrator.train_type_classifier(part="wrist")
    orchestrator.save_model(wrist_detector, "models/wrist_error_type_detector.pkl") # type: ignore # ignore

    orchestrator.plot_feature_importance(wrist_detector, top_n=15)
    orchestrator.confusion_matrix(
        y_true=y_test,
        y_pred=wrist_detector.predict(X_test),
        labels=[0, 1]
    )