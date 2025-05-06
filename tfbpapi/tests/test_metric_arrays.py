import logging

import numpy as np
import pandas as pd
import pytest

from tfbpapi.metric_arrays import metric_arrays


def test_metric_arrays_expected_result(caplog):
    res_dict = {
        "metadata": pd.DataFrame(
            {
                "id": ["A", "B"],
                "regulator_symbol": ["tf1", "tf2"],
            }
        ),
        "data": {
            "A": pd.DataFrame(
                {
                    "target_symbol": ["gene1", "gene2"],
                    "metric1": [1.0, 2.0],
                }
            ),
            "B": pd.DataFrame(
                {
                    "target_symbol": ["gene2", "gene1"],
                    "metric1": [3.0, 4.0],
                }
            ),
        },
    }
    metrics_dict = {"metric1": np.mean}

    # Run function
    with caplog.at_level(logging.WARNING):
        output_dict = metric_arrays(res_dict, metrics_dict)

    # Check expected result for metric1
    # order based on the index of output_dict['metrics1'] since the ordering of
    # the rows is random due to the set operation
    expected_df = pd.DataFrame(
        {"tf1": [1.0, 2.0], "tf2": [4.0, 3.0]},
        index=pd.Index(["gene1", "gene2"], name="target_symbol"),
    ).reindex(output_dict["metric1"].index)

    pd.testing.assert_frame_equal(output_dict["metric1"], expected_df)

    # Check no warning since there are no incomplete rows or columns
    assert "incomplete" not in caplog.text


def test_metric_arrays_missing_data(caplog):
    res_dict = {
        "metadata": pd.DataFrame(
            {
                "id": ["A", "B"],
                "regulator_symbol": ["tf1", "tf2"],
            }
        ),
        "data": {
            "A": pd.DataFrame(
                {
                    "target_symbol": ["gene1", "gene2"],
                    "metric1": [1.0, 2.0],
                }
            ),
            "B": pd.DataFrame(
                {
                    "target_symbol": ["gene1", "gene3"],
                    "metric1": [5.0, 3.0],
                }
            ),
        },
    }
    metrics_dict = {"metric1": np.mean}

    # Run function with incomplete row dropping
    with caplog.at_level(logging.WARNING):
        output_dict1 = metric_arrays(res_dict, metrics_dict, drop_incomplete_rows=False)

    # Check result for metric1 with "gene2" dropped due to missing data in B
    # sort based on output_dict['metric1'] index since
    # the ordering of the rows is random
    expected_df1 = pd.DataFrame(
        {"tf1": [1.0, 2.0, np.nan], "tf2": [5.0, np.nan, 3.0]},
        index=pd.Index(["gene1", "gene2", "gene3"], name="target_symbol"),
    ).reindex(output_dict1["metric1"].index)

    pd.testing.assert_frame_equal(output_dict1["metric1"], expected_df1)

    # Run function with incomplete row dropping
    with caplog.at_level(logging.WARNING):
        output_dict2 = metric_arrays(res_dict, metrics_dict, drop_incomplete_rows=True)

    # Check result for metric1 with "gene2" dropped due to missing data in B
    expected_df2 = pd.DataFrame(
        {"tf1": [1.0], "tf2": [5.0]},
        index=pd.Index(["gene1"], name="target_symbol"),
    ).reindex(output_dict2["metric1"].index)

    pd.testing.assert_frame_equal(output_dict2["metric1"], expected_df2)

    # Check warning for incomplete rows
    assert "2 rows and 0 columns with incomplete records were dropped" in caplog.text


def test_metric_arrays_missing_keys():
    res_dict = {
        "metadata": pd.DataFrame(
            {"id": ["A"], "target_symbol": ["gene1"], "regulator_symbol": ["tf1"]}
        ),
        # Missing data for id "A"
        "data": {},
    }
    metrics_dict = {"metric1": np.mean}

    # Expect a KeyError for missing data keys
    with pytest.raises(KeyError, match="Data dictionary must have the same keys"):
        metric_arrays(res_dict, metrics_dict)


def test_metric_arrays_non_dataframe_value():
    res_dict = {
        "metadata": pd.DataFrame(
            {"id": ["A"], "target_symbol": ["gene1"], "regulator_symbol": ["tf1"]}
        ),
        "data": {"A": [1, 2, 3]},  # Invalid non-DataFrame entry
    }
    metrics_dict = {"metric1": np.mean}

    # Expect ValueError when data dictionary values are not DataFrames
    with pytest.raises(
        ValueError, match="All values in the data dictionary must be DataFrames"
    ):
        metric_arrays(res_dict, metrics_dict)


def test_metric_arrays_duplicate_rows_without_dedup_func():
    res_dict = {
        "metadata": pd.DataFrame(
            {
                "id": ["A"],
                "target_symbol": ["gene1"],
                "regulator_symbol": ["tf1"],
            }
        ),
        "data": {
            "A": pd.DataFrame(
                {
                    "target_symbol": ["gene1", "gene1"],
                    "metric1": [1.0, 2.0],
                }
            ),
        },
    }
    metrics_dict = {"metric1": None}  # No deduplication function provided

    # Expect a ValueError due to duplicate rows without deduplication function
    #
    with pytest.raises(
        ValueError, match="Duplicate entries found for metric 'metric1'"
    ):
        metric_arrays(res_dict, metrics_dict)  # type: ignore


def test_metric_arrays_deduplication_function():
    res_dict = {
        "metadata": pd.DataFrame(
            {
                "id": ["A"],
                "target_symbol": ["gene1"],
                "regulator_symbol": ["tf1"],
            }
        ),
        "data": {
            "A": pd.DataFrame(
                {
                    "target_symbol": ["gene1", "gene1"],
                    "metric1": [1.0, 2.0],
                }
            ),
        },
    }
    metrics_dict = {"metric1": np.mean}  # Deduplication function to average duplicates

    # Run function with deduplication
    output_dict = metric_arrays(res_dict, metrics_dict)

    # Check that duplicates were averaged correctly
    expected_df = pd.DataFrame(
        {"tf1": [1.5]}, pd.Index(["gene1"], name="target_symbol")
    )
    pd.testing.assert_frame_equal(output_dict["metric1"], expected_df)
