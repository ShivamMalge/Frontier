"""Market data endpoints."""

from __future__ import annotations


def test_prices_returns_a_shaped_frame(client, window):
    body = client.post("/api/v1/market/prices", json={"tickers": ["AAA", "BBB"], **window}).json()

    assert body["tickers"] == ["AAA", "BBB"]
    assert body["observations"] == len(body["prices"]["index"])
    assert body["prices"]["columns"] == ["AAA", "BBB"]
    assert all(len(row) == 2 for row in body["prices"]["data"])
    assert body["cached"] is False


def test_second_identical_request_is_served_from_cache(client, window):
    payload = {"tickers": ["AAA"], **window}
    assert client.post("/api/v1/market/prices", json=payload).json()["cached"] is False
    assert client.post("/api/v1/market/prices", json=payload).json()["cached"] is True


def test_unknown_tickers_are_reported_not_fatal(client, window):
    body = client.post("/api/v1/market/prices", json={"tickers": ["AAA", "NOPE"], **window}).json()
    assert body["tickers"] == ["AAA"]
    assert [f["ticker"] for f in body["failed"]] == ["NOPE"]


def test_all_tickers_unknown_is_a_gateway_error(client, window):
    response = client.post("/api/v1/market/prices", json={"tickers": ["NOPE"], **window})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "no_market_data"


def test_returns_are_on_a_returns_scale(client, window):
    body = client.post("/api/v1/market/returns", json={"tickers": ["AAA", "BBB"], **window}).json()
    values = [abs(v) for row in body["returns"]["data"] for v in row if v is not None]
    # Daily simple returns sit within a few percent; price levels would be ~100.
    assert max(values) < 0.5


def test_reversed_window_is_rejected(client):
    response = client.post(
        "/api/v1/market/prices",
        json={"tickers": ["AAA"], "start": "2021-01-01", "end": "2020-01-01"},
    )
    assert response.status_code == 422


def test_lowercase_tickers_are_normalised_and_deduped(client, window):
    body = client.post(
        "/api/v1/market/prices", json={"tickers": ["aaa", "AAA", "bbb"], **window}
    ).json()
    assert body["tickers"] == ["AAA", "BBB"]
