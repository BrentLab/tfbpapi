import os
import pytest

from tfbpapi import HfHu2007Reimand2010API


@pytest.fixture()
def api(tmp_path):
    cache_dir = tmp_path / "hf_cache"
    local_dir = tmp_path / "hf_local"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Token is optional for public datasets; if user has HF_TOKEN env it will be used.
    return HfHu2007Reimand2010API(
        cache_dir=str(cache_dir),
        local_dir=str(local_dir),
    )


def test_retrieval_schema_and_nonempty(api):
    result = api.read(params={})
    table = result["data"]

    # Verify expected columns
    expected_cols = {
        "regulator_locus_tag",
        "regulator_symbol",
        "target_locus_tag",
        "target_symbol",
        "effect",
        "pval",
    }
    assert expected_cols.issubset(set(table.schema.names))

    # Non-empty
    assert table.num_rows > 0


def test_filter_by_symbols_and_ranges(api):
    # Choose a small set of regulators and thresholds
    params = {
        "regulator_symbol": ["HAP4", "GCR1", "ACE2", "HAP1"],
        "effect_min": 1.0,
        "pval_max": 0.05,
    }
    table = api.read(params=params)["data"]

    # Validate constraints hold in the materialized result
    df = table.to_pandas()
    assert not df.empty
    assert set(df["regulator_symbol"]).issubset(set(params["regulator_symbol"]))
    assert (df["effect"] >= params["effect_min"]).all()
    assert (df["pval"] <= params["pval_max"]).all()


def test_memoization_cache_returns_same_object(api):
    params = {"regulator_symbol": ["HAP4"], "effect_min": 0.5}
    result1 = api.read(params=params)
    result2 = api.read(params=params)

    # The in-memory memoization returns the same dict instance
    assert result1 is result2


def test_metadata_best_effort(api):
    result = api.read(params={})
    meta = result["metadata"]
    # Metadata may vary depending on HF API responses; just assert type/keys best-effort
    if meta is not None:
        assert isinstance(meta, dict)
        assert any(k in meta for k in ("filename", "etag", "commit_hash", "location")) 