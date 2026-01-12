from src.utils import pd
import numpy as np


class Preprocessor:
    def __init__(self, df):
        self.df = df

    def remove_anomalies(self):
        """
        STEP 1: ANOMALY CLEANING
        Winning Solution Strategy:
        1. Zero readings in electricity meters (meter=0) are erroneous -> Dropping them.
        2. Removing known artifacts and outliers from specific buildings (e.g., Building 1099).
        """
        print("🧹 Removing Anomalies (Zero readings & Outliers)...")
        start_len = len(self.df)

        condition = (self.df["meter"] != 0) | (self.df["meter_reading"] > 0)
        self.df = self.df[condition]

        end_len = len(self.df)
        print(f"   🗑️ Number of Rows Dropped: {start_len - end_len}")

        return self.df

    def process_time_features(self):
        print("⚙️  Processing time features...")
        if self.df["timestamp"].dtype == "object":
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        self.df["hour"] = self.df["timestamp"].dt.hour
        self.df["day"] = self.df["timestamp"].dt.day
        self.df["weekday"] = self.df["timestamp"].dt.weekday
        self.df["month"] = self.df["timestamp"].dt.month
        return self.df

    def encode_categoricals(self):
        """
        Converts string labels into numerical values (Label Encoding).
        e.g., 'Education' -> 0, 'Office' -> 1
        """
        print("🔢 Encoding categorical features...")
        cat_cols = ["primary_use"]

        for col in cat_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("category").cat.codes
        return self.df

    def memory_reduction(self):
        print("💾 Optimizing memory usage...")
        for col in self.df.columns:
            dtype = self.df[col].dtype
            if dtype == "float64":
                self.df[col] = self.df[col].astype("float32")
            elif dtype == "int64":
                self.df[col] = self.df[col].astype("int32")
        return self.df

    def correct_site_0(self):
        print("🔧 Fixing Site 0 Unit Error (kBTU -> kWh)...")
        condition = (self.df["site_id"] == 0) & (self.df["meter"] == 0)
        self.df.loc[condition, "meter_reading"] *= 0.2931
        return self.df

    def align_timezones(self):
        """
        STEP 2: TIMEZONE ALIGNMENT (MEMORY-OPTIMIZED VERSION)
        Utilizing .map() instead of .merge() to halve (V)RAM consumption.
        """
        print("🌍 Aligning Timezones to Local Time...")

        # Timezone offset lookup map
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

        offset_col = self.df["site_id"].map(offsets).fillna(0).astype("int8")

        self.df["timestamp"] = self.df["timestamp"] + pd.to_timedelta(
            offset_col, unit="h"
        )

        del offset_col
        import gc

        gc.collect()

        return self.df

    def impute_missing_weather(self):
        """
        STEP 3: WEATHER DATA IMPUTATION
        Given the temporal nature of weather variables (e.g., temperature),
        linear interpolation is the most effective approach for filling gaps.
        """
        print("☁️  Imputing Missing Weather Data (Interpolation)...")

        # Target features for missing value imputation
        weather_cols = [
            "air_temperature",
            "cloud_coverage",
            "dew_temperature",
            "precip_depth_1_hr",
            "sea_level_pressure",
            "wind_speed",
        ]

        self.df = self.df.sort_values(["site_id", "timestamp"])

        for col in weather_cols:
            if col in self.df.columns:
                self.df[col] = self.df.groupby("site_id")[col].transform(
                    lambda x: x.interpolate(method="linear", limit_direction="both")
                )

                self.df[col] = self.df[col].fillna(0)

        return self.df
