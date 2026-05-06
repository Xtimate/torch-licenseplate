import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


def make_image_bytes(client, width=200, height=100, color=(255, 220, 0)):
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "components" in data


def test_recognize_returns_result(client):
    img_bytes = make_image_bytes(client)
    res = client.post(
        "/recognize", files={"file": ("plate.jpg", img_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert "text" in data or "reason" in data


def test_recognize_rejected_tiny_image(client):
    img_bytes = make_image_bytes(client, width=1, height=1)
    res = client.post(
        "/recognize", files={"file": ("tiny.jpg", img_bytes, "image/jpeg")}
    )
    assert res.status_code == 200


def test_pipeline_returns_list(client):
    img_bytes = make_image_bytes(client)
    res = client.post("/pipeline", files={"file": ("img.jpg", img_bytes, "image/jpeg")})
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_pipeline_empty_image(client):
    img_bytes = make_image_bytes(client, width=640, height=480, color=(128, 128, 128))
    res = client.post(
        "/pipeline", files={"file": ("empty.jpg", img_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_watchlist_add_and_get(client):
    res = client.post("/watchlist", json={"text": "TESTPLATE", "notes": "pytest"})
    assert res.status_code == 200

    res = client.get("/watchlist")
    assert res.status_code == 200
    texts = [w["text"] for w in res.json()]
    assert "TESTPLATE" in texts


def test_watchlist_delete(client):
    client.post("/watchlist", json={"text": "DELETETEST", "notes": None})
    res = client.delete("/watchlist/DELETETEST")
    assert res.status_code == 200

    res = client.get("/watchlist")
    texts = [w["text"] for w in res.json()]
    assert "DELETETEST" not in texts


def test_history_returns_list(client):
    res = client.get("/history")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_stats_shape(client):
    res = client.get("/stats")
    assert res.status_code == 200
    data = res.json()
    assert "total" in data
    assert "by_country" in data
    assert "by_source" in data


def test_analytics_shape(client):
    res = client.get("/analytics")
    assert res.status_code == 200
    data = res.json()
    assert "recent_24h" in data
    assert "avg_confidence" in data
    assert "watchlist_hits" in data
    assert "confidence_buckets" in data
