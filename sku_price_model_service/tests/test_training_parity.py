"""The API must build exactly the features the model was trained on."""

import importlib.util
from pathlib import Path

import pandas as pd

from app.config import CATEGORICAL_FEATURES, FEATURES
from app.db import ReferenceData
from app.preprocessing import build_features
from src.models.train_model import DEFAULT_DATA_PATH, load_training_data

SEED_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "seed_demo_db.py"


def load_seed_module():
    spec = importlib.util.spec_from_file_location("seed_demo_db", SEED_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_service_features_match_training_features():
    seed = load_seed_module()
    references = ReferenceData.from_frames(**seed.build_reference_tables(pd.read_csv(DEFAULT_DATA_PATH)))
    training = load_training_data(DEFAULT_DATA_PATH)
    request = training[["dates", "SKU", "price"]].rename(columns={"price": "price_per_sku"})

    served = build_features(request, references)
    expected = training[list(FEATURES)]

    numeric = [column for column in FEATURES if column not in CATEGORICAL_FEATURES]
    pd.testing.assert_frame_equal(served[numeric], expected[numeric])
    for column in CATEGORICAL_FEATURES:
        pd.testing.assert_series_equal(served[column].astype(str), expected[column].astype(str))
