import mlflow
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
import numpy as np
import time
import gc

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Ashrae_Energy_Prediction")


def train_model():
    loader = DataLoader()
    df = loader.load_dataset()

    processor = Preprocessor(df)

    df = processor.correct_site_0()
    df = processor.remove_anomalies()
    df = processor.align_timezones()
    df = processor.impute_missing_weather()
    df = processor.process_time_features()
    df = processor.encode_categoricals()
    df = processor.memory_reduction()

    target = "meter_reading"
    features = [col for col in df.columns if col not in [target, "timestamp", "row_id"]]

    print(f"🚀 Training with {len(features)} features...")

    X = df[features]
    y = df[target]

    del df
    gc.collect()

    y = np.log1p(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        params = {
            "objective": "reg:squarederror",
            "device": "cuda",
            "tree_method": "hist",
            "eval_metric": "rmse",
            "learning_rate": 0.1,
            "max_depth": 8,
            "n_estimators": 1000,
            "early_stopping_rounds": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

        mlflow.log_params(params)

        print("🏋️ Model training started using GPU compute backend...")
        start_time = time.time()

        model = xgb.XGBRegressor(**params)

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

        duration = time.time() - start_time
        print(f"✅ Model training successful. Total elapsed time: {duration:.2f}s")

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))

        print(f"📊 Validation RMSE: {rmse:.4f}")

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("training_time", duration)

        mlflow.xgboost.log_model(model, "xgboost_model")
        print("💾 Model successfully logged to MLflow.")


if __name__ == "__main__":
    train_model()
