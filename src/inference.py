import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow


def load_static_data():
    """
    Loads static data (Building & Weather) for the test set once.
    AND PERFORMS MISSING VALUE IMPUTATION.
    """
    print("📥 Loading auxiliary tables (Building & Weather)...")
    building = pd.read_csv("data/raw/building_metadata.csv")
    weather = pd.read_csv("data/raw/weather_test.csv")

    # Timestamp alignment and formatting
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])

    # --- WEATHER DATA IMPUTATION ---
    print("☁️  Executing linear interpolation on weather features...")
    weather_cols = [
        "air_temperature",
        "cloud_coverage",
        "dew_temperature",
        "precip_depth_1_hr",
        "sea_level_pressure",
        "wind_speed",
    ]

    # Perform site-specific linear interpolation
    weather = weather.sort_values(["site_id", "timestamp"])

    for col in weather_cols:
        if col in weather.columns:
            # (Ensures missing values are filled based on local weather trends)
            weather[col] = weather.groupby("site_id")[col].transform(
                lambda x: x.interpolate(method="linear", limit_direction="both")
            )
            # Zero-fill remaining missing values to ensure data integrity
            weather[col] = weather[col].fillna(0)

    # --- MEMORY OPTIMIZATION ---
    for col in weather.columns:
        if weather[col].dtype == "float64":
            weather[col] = weather[col].astype("float32")

    return building, weather


def predict_batch(model, batch_df, building, weather):
    """
    Generates predictions (inference) on a small data subset.
    """
    # Step 1: Merging features with metadata
    batch_df["timestamp"] = pd.to_datetime(batch_df["timestamp"])

    df = batch_df.merge(building, on="building_id", how="left")
    df = df.merge(weather, on=["site_id", "timestamp"], how="left")

    # --- TIMEZONE ALIGNMENT (Consistent with Training Logic) ---
    offsets = {
        0: -5,
        1: 0,
        2: -7,
        3: -5,
        4: -8,
        5: 0,
        6: -5,
        7: -5,
        8: -5,
        9: -6,
        10: -7,
        11: -5,
        12: 0,
        13: -6,
        14: -5,
        15: -5,
    }

    # Merge yerine MAP kullanımı (Hızlı ve RAM dostu)
    offset_values = df["site_id"].map(offsets).fillna(0)
    df["timestamp"] = df["timestamp"] + pd.to_timedelta(offset_values, unit="h")
    # --------------------------------------------------------

    # 2. Feature Engineering (Yerel saate göre)
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month

    if "primary_use" in df.columns:
        df["primary_use"] = df["primary_use"].astype("category").cat.codes

    # Feature Seçimi (Eğitimdeki sırayla AYNI OLMALI)
    features = [
        "building_id",
        "meter",
        "site_id",
        "primary_use",
        "square_feet",
        "year_built",
        "floor_count",
        "air_temperature",
        "cloud_coverage",
        "dew_temperature",
        "precip_depth_1_hr",
        "sea_level_pressure",
        "wind_direction",
        "wind_speed",
        "hour",
        "day",
        "weekday",
        "month",
    ]

    # Check for missing features to prevent runtime errors
    valid_features = [f for f in features if f in df.columns]
    X = df[valid_features]

    # 3. Prediction
    preds = model.predict(X)

    # Step 4: Inverse Log Transformation (Reverting to Original Scale)
    # Converting log-transformed predictions back to actual units
    preds = np.expm1(preds)
    preds = np.maximum(preds, 0)

    return preds
