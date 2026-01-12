import pandas as pd
import mlflow
import gc
import os
import numpy as np
from tqdm import tqdm
from src.inference import load_static_data, predict_batch

CHUNK_SIZE = 500_000
OUTPUT_FILE = "submission_lgbm.csv"


def main():
    print("🚀 Launching LightGBM prediction pipeline...")

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    print("🧠 Loading LightGBM (Production stage) via MLflow...")
    try:
        last_run = mlflow.search_runs(
            experiment_names=["Ashrae_Energy_Prediction"], order_by=["start_time DESC"]
        ).iloc[0]
        run_id = last_run.run_id
        print(f"   Bulunan Son Run ID: {run_id}")
        model_uri = f"runs:/{run_id}/lightgbm_model"
        loaded_model = mlflow.lightgbm.load_model(model_uri)
        print("   ✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Model not found: {e}")
        return

    building, weather = load_static_data()

    print("⏳ Initiating Batch Inference...")

    total_rows = 41697600
    total_chunks = (total_rows // CHUNK_SIZE) + 1

    test_iterator = pd.read_csv("data/raw/test.csv", chunksize=CHUNK_SIZE)

    first_chunk = True

    for i, batch_df in tqdm(enumerate(test_iterator), total=total_chunks, unit="chunk"):
        row_ids = batch_df["row_id"].values
        preds = predict_batch(loaded_model, batch_df, building, weather)
        submission_chunk = pd.DataFrame({"row_id": row_ids, "meter_reading": preds})

        if first_chunk:
            submission_chunk.to_csv(OUTPUT_FILE, mode="w", header=True, index=False)
            first_chunk = False
        else:
            submission_chunk.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)

        del batch_df, submission_chunk, preds
        gc.collect()

    print(f"\n✅ PROCESS COMPLETED! Submission file is ready at: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
