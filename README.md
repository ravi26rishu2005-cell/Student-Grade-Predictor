# Student-Grade-Predictor
The **Student Grade Predictor** is a machine learning–based system that predicts a student’s academic performance using factors such as attendance, study time, assignments, and test scores. It helps students understand their expected grades and identify weak areas so they can improve their performance before final exams through better planning.


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
import os
import warnings

warnings.filterwarnings('ignore')


# =========================
# DATA PROCESSING CLASS
# =========================
class DataProcessor:

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None

    def load_data(self, file_path='data/student_data.csv'):
        """Load the student dataset"""
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                return df
            else:
                return self._create_synthetic_data()
        except Exception as e:
            print(f"Error loading data: {e}")
            return self._create_synthetic_data()

    def _create_synthetic_data(self):
        """Create synthetic dataset"""
        np.random.seed(42)

        n_samples = 395

        data = {
            "attendance": np.random.randint(50, 100, n_samples),
            "study_time": np.random.randint(1, 5, n_samples),
            "assignments": np.random.randint(40, 100, n_samples),
            "test_scores": np.random.randint(30, 100, n_samples),
            "final_grade": np.random.randint(0, 20, n_samples)
        }

        df = pd.DataFrame(data)
        return df


# =========================
# MODEL TRAINER CLASS
# =========================
class ModelTrainer:

    def __init__(self):
        self.data_processor = DataProcessor()
        self.models = {}
        self.metrics = {}

    def prepare_data(self):

        df = self.data_processor.load_data()

        X = df.drop("final_grade", axis=1)
        y = df["final_grade"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test

    def train_random_forest(self, X_train, X_test, y_train, y_test):

        rf_reg = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        rf_reg.fit(X_train, y_train)

        predictions = rf_reg.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        self.models["RandomForest"] = rf_reg

        self.metrics["RandomForest"] = {
            "MSE": mse,
            "MAE": mae,
            "R2 Score": r2
        }

        return predictions

    def train_xgboost(self, X_train, X_test, y_train, y_test):

        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )

        xgb_model.fit(X_train, y_train)

        predictions = xgb_model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        self.models["XGBoost"] = xgb_model

        self.metrics["XGBoost"] = {
            "MSE": mse,
            "MAE": mae,
            "R2 Score": r2
        }

        return predictions


# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":

    trainer = ModelTrainer()

    X_train, X_test, y_train, y_test = trainer.prepare_data()

    trainer.train_random_forest(X_train, X_test, y_train, y_test)

    trainer.train_xgboost(X_train, X_test, y_train, y_test)

    print("Model Performance Metrics:")
    print(trainer.metrics)


        
