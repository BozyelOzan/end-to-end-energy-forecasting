import pytest
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.utils import pd


def test_data_loader_structure():
    loader = DataLoader()

    df = loader.generate_synthetic_data(n_rows=100)

    assert len(df) == 100
    assert "building_id" in df.columns
    assert "meter_reading" in df.columns


def test_preprocessing_logic():
    loader = DataLoader()
    df = loader.generate_synthetic_data(n_rows=50)

    processor = Preprocessor(df)
    df = processor.process_time_features()

    assert "hour" in df.columns
    assert "month" in df.columns

    assert df["month"].min() >= 1
    assert df["month"].max() <= 12
