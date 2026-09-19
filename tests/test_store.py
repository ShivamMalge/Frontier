"""The Parquet price store.

The property that matters is reproducibility. yfinance restates history, so a
pipeline that re-downloads cannot repeat its own results. These tests pin down that
the store is idempotent, grows incrementally, and never changes a stored bar without
saying so.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from Layer1_Preprocessing.frames import pandas_wide_to_polars, wide_to_long
from Layer1_Preprocessing.store import PriceStore
from Layer1_Preprocessing.synthetic import synthetic_prices

TICKERS = ["AAA", "BBB", "CCC"]


def ohlcv(tickers: list[str], start: str, end: str) -> pl.DataFrame:
    """Synthetic OHLCV in the store's long schema."""
    frame = wide_to_long(pandas_wide_to_polars(synthetic_prices(tickers, start, end)))
    return frame.rename({"price": "adj_close"}).with_columns(
        pl.col("adj_close").alias("open"),
        (pl.col("adj_close") * 1.01).alias("high"),
        (pl.col("adj_close") * 0.99).alias("low"),
        pl.col("adj_close").alias("close"),
        pl.lit(1_000_000).cast(pl.Int64).alias("volume"),
    )


@pytest.fixture
def store(tmp_path) -> PriceStore:
    return PriceStore(tmp_path / "store")


@pytest.fixture
def populated(store) -> PriceStore:
    store.write(ohlcv(TICKERS, "2020-01-01", "2021-01-01"), dt.datetime(2026, 1, 1, 9))
    return store


class TestLayout:
    def test_empty_store_reports_itself_empty(self, store):
        assert not store.exists()
        assert store.stored_tickers() == []
        assert store.read().is_empty()

    def test_hive_partitioned_one_file_per_ticker(self, populated):
        files = sorted(
            path.relative_to(populated.root).as_posix()
            for path in populated.root.rglob("*.parquet")
        )
        assert files == [f"prices/ticker={t}/data.parquet" for t in TICKERS]

    def test_partition_column_is_not_duplicated_inside_the_file(self, populated):
        """The ticker lives in the directory name; storing it twice wastes space."""
        raw = pl.read_parquet(populated.ticker_path("AAA"))
        assert "ticker" not in raw.columns
        # It reappears on read, from the partition.
        assert "ticker" in populated.read(["AAA"]).columns

    def test_stored_tickers_are_discovered_from_the_layout(self, populated):
        assert populated.stored_tickers() == TICKERS


class TestIdempotence:
    def test_writing_the_same_window_twice_adds_nothing(self, store):
        frame = ohlcv(TICKERS, "2020-01-01", "2020-06-01")
        first = store.write(frame, dt.datetime(2026, 1, 1, 9))
        second = store.write(frame, dt.datetime(2026, 1, 2, 9))

        assert first.rows_added > 0
        assert second.rows_added == 0
        assert second.rows_revised == 0
        assert store.read().height == first.rows_written

    def test_extending_the_window_adds_only_the_new_dates(self, store):
        store.write(ohlcv(TICKERS, "2020-01-01", "2020-06-01"), dt.datetime(2026, 1, 1, 9))
        before = store.read().height

        report = store.write(ohlcv(TICKERS, "2020-01-01", "2020-09-01"), dt.datetime(2026, 1, 2, 9))
        assert report.rows_revised == 0
        assert report.rows_added > 0
        assert store.read().height == before + report.rows_added

    def test_a_new_ticker_leaves_the_others_untouched(self, populated):
        before = populated.read(["AAA"])
        populated.write(ohlcv(["DDD"], "2020-01-01", "2021-01-01"), dt.datetime(2026, 2, 1, 9))

        assert populated.stored_tickers() == [*TICKERS, "DDD"]
        assert populated.read(["AAA"]).equals(before)


class TestRevisions:
    def test_an_upstream_restatement_is_detected_and_reported(self, populated):
        """The reason the store exists.

        A split adjustment restates every bar before the split. Overwriting silently
        would mean a backtest run yesterday cannot be reproduced today, with nothing
        in the record to explain why.
        """
        restated = ohlcv(TICKERS, "2020-01-01", "2021-01-01").with_columns(
            pl.when((pl.col("ticker") == "AAA") & (pl.col("date") < dt.date(2020, 3, 1)))
            .then(pl.col("adj_close") / 2.0)
            .otherwise(pl.col("adj_close"))
            .alias("adj_close")
        )
        report = populated.write(restated, dt.datetime(2026, 3, 1, 9))

        assert report.rows_revised > 0
        assert report.revised_tickers == ["AAA"]
        assert report.rows_added == 0

        first = report.revisions.row(0, named=True)
        assert first["ticker"] == "AAA"
        assert first["new_adj_close"] == pytest.approx(first["old_adj_close"] / 2.0)

    def test_the_new_value_wins(self, populated):
        restated = ohlcv(["AAA"], "2020-01-01", "2021-01-01").with_columns(
            (pl.col("adj_close") * 2.0).alias("adj_close")
        )
        populated.write(restated, dt.datetime(2026, 3, 1, 9))

        stored = populated.read(["AAA"]).sort("date")["adj_close"].to_numpy()
        expected = restated.sort("date")["adj_close"].to_numpy()
        assert stored == pytest.approx(expected)

    def test_floating_point_noise_is_not_reported_as_a_revision(self, populated):
        """A 1e-15 difference is a round trip, not a restatement."""
        jittered = ohlcv(TICKERS, "2020-01-01", "2021-01-01").with_columns(
            (pl.col("adj_close") * (1.0 + 1e-15)).alias("adj_close")
        )
        assert populated.write(jittered, dt.datetime(2026, 3, 1, 9)).rows_revised == 0

    def test_vintages_accumulate(self, store):
        store.write(ohlcv(["AAA"], "2020-01-01", "2020-06-01"), dt.datetime(2026, 1, 1, 9))
        store.write(ohlcv(["AAA"], "2020-06-01", "2021-01-01"), dt.datetime(2026, 2, 1, 9))

        vintages = store.vintages()
        assert vintages.height == 2
        assert int(vintages["rows"].sum()) == store.read().height


class TestReads:
    def test_ticker_filter(self, populated):
        assert populated.read(["AAA"])["ticker"].unique().to_list() == ["AAA"]

    def test_date_range_is_half_open(self, populated):
        frame = populated.read(["AAA"], dt.date(2020, 3, 1), dt.date(2020, 4, 1))
        assert frame["date"].min() >= dt.date(2020, 3, 1)
        # `end` is exclusive, matching the download and the rest of the API.
        assert frame["date"].max() < dt.date(2020, 4, 1)

    def test_wide_read_has_one_column_per_ticker(self, populated):
        wide = populated.read_wide(["AAA", "BBB"])
        assert wide.columns == ["date", "AAA", "BBB"]
        assert wide.height == populated.read(["AAA"]).height

    def test_wide_read_can_select_any_stored_column(self, populated):
        volume = populated.read_wide(["AAA"], column="volume")
        assert volume["AAA"].to_list() == [1_000_000] * volume.height

    def test_unknown_ticker_reads_empty_rather_than_raising(self, populated):
        assert populated.read(["NOPE"]).is_empty()


class TestDuckDbQueries:
    def test_coverage_summarises_each_ticker(self, populated):
        coverage = populated.coverage()
        assert coverage["ticker"].to_list() == TICKERS
        assert (coverage["rows"] > 0).all()
        assert (coverage["first_date"] < coverage["last_date"]).all()
        assert coverage["vintages"].to_list() == [1, 1, 1]

    def test_gaps_finds_a_hole_and_ignores_weekends(self, store):
        frame = ohlcv(["AAA"], "2020-01-01", "2020-06-01")
        # Remove a fortnight.
        holed = frame.filter(~pl.col("date").is_between(dt.date(2020, 3, 2), dt.date(2020, 3, 16)))
        store.write(holed, dt.datetime(2026, 1, 1, 9))

        gaps = store.gaps(max_gap_days=5)
        assert gaps.height == 1
        assert gaps["gap_days"][0] > 5
        # A 3-day weekend gap must not be reported at this threshold.
        assert store.gaps(max_gap_days=30).is_empty()

    def test_arbitrary_sql_against_the_prices_view(self, populated):
        result = populated.sql(
            "SELECT ticker, AVG(adj_close) AS mean FROM prices GROUP BY ticker ORDER BY ticker"
        )
        assert result["ticker"].to_list() == TICKERS
        assert (result["mean"] > 0).all()

    def test_sql_on_an_empty_store_returns_no_rows_rather_than_failing(self, store):
        assert store.sql("SELECT * FROM prices").is_empty()
        assert store.coverage().is_empty()


class TestPolarsAndDuckDbAgree:
    def test_both_engines_read_the_same_numbers(self, populated):
        """They overlap deliberately; if they disagreed, one of them is wrong."""
        polars_rows = populated.read(["AAA"]).sort("date").select("date", "adj_close")
        duckdb_rows = populated.sql(
            "SELECT date, adj_close FROM prices WHERE ticker = 'AAA' ORDER BY date"
        )
        assert duckdb_rows["adj_close"].to_numpy() == pytest.approx(
            polars_rows["adj_close"].to_numpy()
        )
        assert duckdb_rows["date"].to_list() == polars_rows["date"].to_list()
