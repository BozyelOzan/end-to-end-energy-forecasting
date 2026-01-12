# Dosya: ensemble.py
import pandas as pd
import mlflow
import gc
import os
import numpy as np
from tqdm import tqdm
from src.inference import load_static_data, predict_batch


XGB_RUN_ID = "51c8e259d63740aba7220e02dfeec6fe"

LGBM_RUN_ID = "04fc5c6dac964b659aba0c7cd659a595"

W_XGB = 0.516
W_LGB = 0.484

CHUNK_SIZE = 500_000
OUTPUT_FILE = "submission_ensemble.csv"


def main():
    print("🚀 Starting weighted ensemble (blending) pipeline...")
    print(f"⚖️  Model Weights -> XGBoost: {W_XGB} | LightGBM: {W_LGB}")

    # 0. Output Directory Cleanup
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    # Step 1: Loading Trained Models (Artifacts)
    print("🧠 Retrieving registered model artifacts from MLflow...")
    try:
        # Step 1.1: Loading Pre-trained XGBoost Model
        xgb_uri = f"runs:/{XGB_RUN_ID}/xgboost_model"
        xgb_model = mlflow.xgboost.load_model(xgb_uri)
        print("   ✅ XGBoost model loaded successfully.")

        # Step 1.2: Loading Pre-trained LightGBM Model
        lgb_uri = f"runs:/{LGBM_RUN_ID}/lightgbm_model"
        lgb_model = mlflow.lightgbm.load_model(lgb_uri)
        print("   ✅ LightGBM model loaded successfully.")

    except Exception as e:
        print(f"❌ Critical: Model retrieval failed. Exception: {e}")
        print(
            "Action Required: Ensure the Run IDs match the existing experiments in the Registry."
        )
        return

    # Step 2: Loading Static Metadata
    building, weather = load_static_data()

    # Step 3: Batch Inference Loop
    print("⏳ Commencing joint inference using XGBoost and LightGBM...")

    total_rows = 41697600
    total_chunks = (total_rows // CHUNK_SIZE) + 1

    test_iterator = pd.read_csv("data/raw/test.csv", chunksize=CHUNK_SIZE)
    first_chunk = True

    for i, batch_df in tqdm(enumerate(test_iterator), total=total_chunks, unit="chunk"):
        row_ids = batch_df["row_id"].values

        pred_xgb = predict_batch(xgb_model, batch_df.copy(), building, weather)
        pred_lgb = predict_batch(lgb_model, batch_df.copy(), building, weather)

        # --- BLENDING ---
        pred_final = (pred_xgb * W_XGB) + (pred_lgb * W_LGB)

        submission_chunk = pd.DataFrame(
            {"row_id": row_ids, "meter_reading": pred_final}
        )

        if first_chunk:
            submission_chunk.to_csv(OUTPUT_FILE, mode="w", header=True, index=False)
            first_chunk = False
        else:
            submission_chunk.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)

        del batch_df, submission_chunk, pred_xgb, pred_lgb, pred_final
        gc.collect()

    print(f"\n✅ Pipeline Finished! Ensemble submission exported to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
