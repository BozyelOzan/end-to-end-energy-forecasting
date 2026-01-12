import os
import gc
import numpy as np
from src.utils import pd  # GPU (cuDF) veya CPU (Pandas)


class DataLoader:
    def __init__(self, data_path="data/raw"):
        self.data_path = data_path

    def load_dataset(self):
        """
        FOR FULL-SCALE TRAINING:
        Loads and integrates Train, Building, and Weather datasets.
        """
        print("📥 Loading datasets (GPU/CPU)...")

        dtype_train = {
            "building_id": "int16",
            "meter": "int8",
            "meter_reading": "float32",
        }

        # File Paths
        train_path = os.path.join(self.data_path, "train.csv")
        building_path = os.path.join(self.data_path, "building_metadata.csv")
        weather_path = os.path.join(self.data_path, "weather_train.csv")

        train = pd.read_csv(train_path, dtype=dtype_train)
        building = pd.read_csv(building_path)
        weather = pd.read_csv(weather_path)

        train["timestamp"] = pd.to_datetime(train["timestamp"])
        weather["timestamp"] = pd.to_datetime(weather["timestamp"])

        print("🔄 Merging tables...")
        df = train.merge(building, on="building_id", how="left")
        df = df.merge(weather, on=["site_id", "timestamp"], how="left")

        del train, building, weather
        gc.collect()

        print(f"✅ Data integration successful. Output dimensions: {df.shape}")
        return df

    def generate_synthetic_data(self, n_rows=1000):
        """
        FOR TESTING PURPOSES (CI/CD):
        Generates mock data to validate pipeline logic
        without loading large-scale production datasets.
        """
        # Small sample size is sufficient for unit/integration tests
        data = {
            "building_id": np.random.randint(0, 100, n_rows),
            "meter": np.random.randint(0, 4, n_rows),
            "timestamp": pd.date_range(
                start="2024-01-01", periods=n_rows, freq="h"
            ),  # 'h' for hourly
            "meter_reading": np.random.uniform(0, 100, n_rows).astype("float32"),
            "site_id": np.random.randint(0, 5, n_rows),
            "primary_use": np.random.choice(["Education", "Office", "Lodging"], n_rows),
            "square_feet": np.random.randint(1000, 10000, n_rows),
            "year_built": np.random.randint(1900, 2020, n_rows),
            "floor_count": np.random.randint(1, 20, n_rows),
            "air_temperature": np.random.uniform(-10, 30, n_rows),
            "cloud_coverage": np.random.randint(0, 9, n_rows),
            "dew_temperature": np.random.uniform(-10, 20, n_rows),
            "precip_depth_1_hr": np.random.uniform(0, 5, n_rows),
            "sea_level_pressure": np.random.uniform(980, 1020, n_rows),
            "wind_direction": np.random.uniform(0, 360, n_rows),
            "wind_speed": np.random.uniform(0, 15, n_rows),
        }

        df = pd.DataFrame(data)

        return df
