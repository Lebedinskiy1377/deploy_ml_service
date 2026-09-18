import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app import main
from app.main import app, get_predictor, get_reference_data

from .factories import reference_frames

REQUEST_CSV = "dates,SKU,price_per_sku,comment\n2019-12-24,101,99.5,\n2019-06-01,202,45,promo test\n"


def upload(client: TestClient, endpoint: str, content: str = REQUEST_CSV, filename: str = "request.csv"):
    return client.post(endpoint, files={"file": (filename, content.encode(), "text/csv")})


@pytest.fixture
def client(reference_data, predictor):
    app.dependency_overrides[get_reference_data] = lambda: reference_data
    app.dependency_overrides[get_predictor] = lambda: predictor
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def sqlite_engine(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'reference.db'}")
    monkeypatch.setattr(main, "get_engine", lambda: engine)
    return engine


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_invocation_returns_request_rows_with_demand(client):
    response = upload(client, "/invocation")

    assert response.status_code == 200
    rows = response.json()
    assert [row["SKU"] for row in rows] == [101, 202]
    assert [row["num_purchases"] for row in rows] == pytest.approx([100.5, 155.0])
    assert rows[0]["comment"] is None  # empty cells come back as null, not NaN
    assert rows[1]["comment"] == "promo test"


def test_optimize_price_returns_base_and_optimal_scenarios(client):
    response = upload(client, "/optimize_price")

    assert response.status_code == 200
    rows = response.json()
    assert [row["dates"] for row in rows] == ["2019-12-24", "2019-06-01"]
    assert [row["price_per_sku"] for row in rows] == [99.5, 45.0]
    expected_columns = {
        "dates",
        "SKU",
        "price_per_sku",
        "cost",
        "base_demand",
        "base_gmv",
        "base_margin",
        "optimal_price",
        "expected_demand",
        "gmv",
        "margin",
        "score",
    }
    assert all(set(row) == expected_columns for row in rows)


@pytest.mark.parametrize(
    ("content", "filename", "message"),
    [
        (REQUEST_CSV, "request.txt", "Only .csv"),
        ("", "request.csv", "empty"),
        ("dates,SKU\n2019-12-24,101\n", "request.csv", "Missing required columns"),
        ("dates,SKU,price_per_sku\n2019-12-24,999,10\n", "request.csv", "Unknown SKU"),
    ],
)
@pytest.mark.parametrize("endpoint", ["/invocation", "/optimize_price"])
def test_bad_requests_are_client_errors(client, endpoint, content, filename, message):
    response = upload(client, endpoint, content, filename)

    assert response.status_code == 400
    assert message in response.json()["detail"]


def test_reads_reference_tables_from_the_database(sqlite_engine, predictor):
    for name, table in reference_frames().items():
        table.to_sql(name, sqlite_engine, index=False)
    app.dependency_overrides[get_predictor] = lambda: predictor
    try:
        response = upload(TestClient(app), "/optimize_price")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert [row["cost"] for row in response.json()] == [80.0, 30.0]


def test_missing_reference_tables_return_503(sqlite_engine, predictor):
    app.dependency_overrides[get_predictor] = lambda: predictor
    try:
        response = upload(TestClient(app), "/invocation")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "seed" in response.json()["detail"]


def test_unavailable_model_returns_503(reference_data, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("Registered model not found")

    monkeypatch.setattr(main, "_predictor", None)
    monkeypatch.setattr(main.DemandPredictor, "from_registry", fail)
    app.dependency_overrides[get_reference_data] = lambda: reference_data
    try:
        response = upload(TestClient(app), "/optimize_price")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "trainer" in response.json()["detail"]
