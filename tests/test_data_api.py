"""Price store endpoints."""

from __future__ import annotations

import datetime as dt
import time

import polars as pl
import pytest
from fastapi.testclient import TestClient

from app import jobs
from app.main import create_app
from app.settings import get_settings
from Layer1_Preprocessing.store import PriceStore
from tests.test_store import TICKERS, ohlcv


@pytest.fixture
def store_client(tmp_path, monkeypatch):
    """A client whose market data comes from a populated Parquet store."""
    root = tmp_path / "store"
    monkeypatch.setenv("SO_DATA_ROOT", str(root))
    monkeypatch.setenv("SO_MARKET_DATA_SOURCE", "parquet")
    get_settings.cache_clear()

    PriceStore(root).write(
        ohlcv(TICKERS, "2018-01-01", "2022-01-01"), dt.datetime(2026, 1, 1, 9)
    )

    jobs.set_job_store(None)
    with TestClient(create_app()) as client:
        yield client
    jobs.set_job_store(None)
    get_settings.cache_clear()


class TestCoverage:
    def test_reports_what_is_stored(self, store_client):
        body = store_client.get("/api/v1/data/coverage").json()
        assert body["exists"] is True
        assert body["tickers"] == TICKERS
        assert body["total_rows"] > 0
        assert {row["ticker"] for row in body["coverage"]} == set(TICKERS)
        for row in body["coverage"]:
            assert row["rows"] > 0
            assert row["first_date"] < row["last_date"]
            assert row["vintages"] == 1

    def test_empty_store_is_reported_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SO_DATA_ROOT", str(tmp_path / "nothing"))
        monkeypatch.setenv("SO_MARKET_DATA_SOURCE", "parquet")
        get_settings.cache_clear()
        jobs.set_job_store(None)
        try:
            with TestClient(create_app()) as client:
                body = client.get("/api/v1/data/coverage").json()
            assert body["exists"] is False
            assert body["tickers"] == []
            assert body["total_rows"] == 0
        finally:
            jobs.set_job_store(None)
            get_settings.cache_clear()

    def test_gaps_are_surfaced(self, tmp_path, monkeypatch):
        root = tmp_path / "store"
        monkeypatch.setenv("SO_DATA_ROOT", str(root))
        monkeypatch.setenv("SO_MARKET_DATA_SOURCE", "parquet")
        get_settings.cache_clear()

        holed = ohlcv(["AAA"], "2020-01-01", "2020-06-01").filter(
            ~pl.col("date").is_between(dt.date(2020, 3, 2), dt.date(2020, 3, 20))
        )
        PriceStore(root).write(holed, dt.datetime(2026, 1, 1, 9))

        jobs.set_job_store(None)
        try:
            with TestClient(create_app()) as client:
                body = client.get("/api/v1/data/coverage?max_gap_days=5").json()
            assert len(body["gaps"]) == 1
            assert body["gaps"][0]["ticker"] == "AAA"
            assert body["gaps"][0]["gap_days"] > 5
        finally:
            jobs.set_job_store(None)
            get_settings.cache_clear()


class TestParquetSource:
    def test_prices_are_served_from_the_store(self, store_client):
        body = store_client.post(
            "/api/v1/market/prices",
            json={"tickers": ["AAA", "BBB"], "start": "2019-01-01", "end": "2020-01-01"},
        ).json()
        assert body["tickers"] == ["AAA", "BBB"]
        assert body["observations"] > 200

    def test_a_ticker_not_in_the_store_is_refused_rather_than_downloaded(
        self, store_client
    ):
        """Falling back to the network would undo the whole point of the store."""
        response = store_client.post(
            "/api/v1/market/prices",
            json={"tickers": ["AAA", "ZZZZ"], "start": "2019-01-01", "end": "2020-01-01"},
        )
        assert response.status_code == 502
        body = response.json()["error"]
        assert body["code"] == "no_market_data"
        assert "ZZZZ" in body["message"]
        assert "never falls back" in body["message"]

    def test_repeated_reads_are_identical(self, store_client):
        """Reproducibility: the same request must give the same numbers."""
        payload = {"tickers": TICKERS, "start": "2019-01-01", "end": "2021-01-01"}

        def fetch():
            return store_client.post("/api/v1/market/prices", json=payload).json()["prices"]

        assert fetch()["data"] == fetch()["data"]

    def test_the_full_pipeline_runs_off_the_store(self, store_client):
        submitted = store_client.post(
            "/api/v1/pipeline/runs",
            json={
                "tickers": TICKERS,
                "backend": "naive",
                "start": "2018-01-01",
                "end": "2022-01-01",
            },
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]

        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            body = store_client.get(f"/api/v1/pipeline/runs/{job_id}/result").json()
            if body["state"] not in {"queued", "running"}:
                break
            time.sleep(0.05)

        assert body["state"] == "succeeded", body["message"]
        assert body["result"]["tickers"] == TICKERS


class TestQuery:
    def test_aggregation_over_the_prices_view(self, store_client):
        body = store_client.post(
            "/api/v1/data/query",
            json={
                "sql": "SELECT ticker, COUNT(*) AS n FROM prices GROUP BY ticker ORDER BY ticker"
            },
        ).json()
        assert body["columns"] == ["ticker", "n"]
        assert [row[0] for row in body["rows"]] == TICKERS
        assert body["truncated"] is False

    def test_dates_are_serialised_as_strings(self, store_client):
        body = store_client.post(
            "/api/v1/data/query",
            json={"sql": "SELECT date, adj_close FROM prices ORDER BY date LIMIT 3"},
        ).json()
        assert all(isinstance(row[0], str) for row in body["rows"])

    def test_results_are_truncated_and_flagged(self, store_client):
        body = store_client.post(
            "/api/v1/data/query", json={"sql": "SELECT * FROM prices", "limit": 5}
        ).json()
        assert body["row_count"] == 5
        assert body["truncated"] is True

    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE prices",
            "CREATE TABLE evil AS SELECT 1",
            "COPY (SELECT 1) TO '/tmp/evil.csv'",
            "ATTACH '/tmp/other.db' AS other",
            "INSTALL httpfs",
            "PRAGMA database_list",
            "DELETE FROM prices",
        ],
    )
    def test_write_and_filesystem_statements_are_refused(self, store_client, sql):
        """DuckDB can reach the local filesystem, so the endpoint is read-only."""
        response = store_client.post("/api/v1/data/query", json={"sql": sql})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "unsafe_query"

    def test_a_select_that_hides_a_second_statement_is_refused(self, store_client):
        response = store_client.post(
            "/api/v1/data/query",
            json={"sql": "SELECT 1; DROP TABLE prices"},
        )
        assert response.status_code == 422

    def test_a_select_containing_a_forbidden_word_in_a_string_is_still_refused(
        self, store_client
    ):
        """Deliberately conservative: a keyword check cannot parse SQL properly.

        Refusing a harmless query is a far better failure than allowing a COPY.
        """
        response = store_client.post(
            "/api/v1/data/query", json={"sql": "SELECT 'create' AS word"}
        )
        assert response.status_code == 422


class TestIngest:
    def test_ingest_is_submitted_as_a_job(self, store_client, monkeypatch):
        """The download is stubbed; the job wiring is what is under test."""
        from app.services import market_data

        def fake_download(tickers, start, end):
            return ohlcv([t for t in tickers if t in (*TICKERS, "DDD")], start, end)

        monkeypatch.setattr(market_data, "download_ohlcv", fake_download)

        submitted = store_client.post(
            "/api/v1/data/ingest",
            json={"tickers": ["DDD"], "start": "2020-01-01", "end": "2021-01-01"},
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]

        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            body = store_client.get(f"/api/v1/pipeline/runs/{job_id}/result").json()
            if body["state"] not in {"queued", "running"}:
                break
            time.sleep(0.05)

        assert body["state"] == "succeeded", body["message"]
        result = body["result"]
        assert result["tickers"] == ["DDD"]
        assert result["rows_added"] > 0
        assert result["rows_revised"] == 0

        assert "DDD" in store_client.get("/api/v1/data/coverage").json()["tickers"]
