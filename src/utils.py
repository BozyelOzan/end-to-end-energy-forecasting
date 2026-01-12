import warnings
import pandas as pd
import numpy as np

"""

def get_dataframe_library(use_gpu=True):

    if use_gpu:
        try:
            import cudf
            cudf.Series([1])
            print("🚀 PERFORMANCE MODE: GPU Acceleration (RAPIDS cuDF) Enabled")
            return cudf
        except (ImportError, Exception) as e:
            warnings.warn(f"⚠️ GPU/RAPIDS not available ({e}). Falling back to Pandas.")

    import pandas as pd

    print("🐢 COMPATIBILITY MODE: CPU (Pandas) Enabled")
    return pd


pd = get_dataframe_library(use_gpu=True)
"""

print("🐢 Compatibility Mode: CPU (Pandas) for ETL -> GPU for Training")
