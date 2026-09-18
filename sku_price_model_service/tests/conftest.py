import pandas as pd
import pytest

from app.db import ReferenceData
from app.demand_predictor import DemandPredictor

from .factories import LinearDemandModel, reference_frames


@pytest.fixture
def reference_data() -> ReferenceData:
    return ReferenceData.from_frames(**reference_frames())


@pytest.fixture
def demand_model() -> LinearDemandModel:
    return LinearDemandModel()


@pytest.fixture
def predictor(demand_model: LinearDemandModel) -> DemandPredictor:
    return DemandPredictor(demand_model, version="test")


@pytest.fixture
def request_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dates": ["2019-12-24", "2019-06-01"],
            "SKU": [101, 202],
            "price_per_sku": [99.5, 45.0],
        }
    )
