import mlflow
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
import numpy as np
import time
import gc


mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Ashrae_Energy_Prediction")


def train_lgbm():
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

    print(f"🚀 Training LightGBM with {len(features)} features...")

    X = df[features]
    y = df[target]

    del df
    gc.collect()

    y = np.log1p(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_features = [
        "site_id",
        "building_id",
        "primary_use",
        "meter",
        "weekday",
        "hour",
        "month",
    ]
    cat_feats = [c for c in categorical_features if c in X_train.columns]

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_feats)
    val_data = lgb.Dataset(
        X_val, label=y_val, reference=train_data, categorical_feature=cat_feats
    )

    with mlflow.start_run(run_name="LightGBM_Model"):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 1280,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.9,
            "bagging_freq": 5,
            "device": "cpu",
            "n_jobs": -1,
            "verbose": -1,
        }

        mlflow.log_params(params)
        print("🏋️ Fitting LightGBM model using CPU backend...")
        start_time = time.time()

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100),
            ],
        )

        duration = time.time() - start_time
        print(f"✅ Model training successful. Total elapsed time: {duration:.2f}s")

        preds = model.predict(X_val, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_val, preds))

        print(f"📊 LightGBM Validation RMSE: {rmse:.4f}")

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("training_time", duration)

        mlflow.lightgbm.log_model(model, "lightgbm_model")
        print("💾 Model artifacts and parameters tracked in MLflow.")


if __name__ == "__main__":
    train_lgbm()
