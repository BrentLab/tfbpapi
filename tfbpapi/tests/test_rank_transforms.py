import numpy as np
from scipy.stats import rankdata

from tfbpapi.rank_transforms import (
    shifted_negative_log_ranks,
    transform,
)


def test_shifted_negative_log_ranks_basic():
    ranks = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected_log_ranks = -1 * np.log10(ranks) + np.log10(np.max(ranks))

    actual_log_ranks = shifted_negative_log_ranks(ranks)
    np.testing.assert_array_almost_equal(actual_log_ranks, expected_log_ranks)


def test_shifted_negative_log_ranks_with_ties():
    ranks = np.array([1.0, 2.5, 2.5, 3.0, 4.0])
    expected_log_ranks = -1 * np.log10(ranks) + np.log10(np.max(ranks))

    actual_log_ranks = shifted_negative_log_ranks(ranks)
    np.testing.assert_array_almost_equal(actual_log_ranks, expected_log_ranks)


def test_negative_log_transform_basic():
    pvalues = np.array([0.01, 0.05, 0.01, 0.02, 0.05])
    enrichment = np.array([5.0, 3.0, 6.0, 4.0, 4.5])

    # Expected ranks based on pvalue (primary) with enrichment (secondary) tie-breaking
    expected_ranks = np.array([2.0, 5.0, 1.0, 3.0, 4.0])
    expected_log_ranks = -1 * np.log10(expected_ranks) + np.log10(
        np.max(expected_ranks)
    )

    actual_log_ranks = transform(pvalues, enrichment)
    np.testing.assert_array_almost_equal(actual_log_ranks, expected_log_ranks)


def test_all_ties_in_primary_column():
    pvalues = np.array([0.01, 0.01, 0.01, 0.01])
    enrichment = np.array([10.0, 20.0, 15.0, 5.0])

    # With all pvalues tied, the ranking should depend solely
    # on enrichment (higher is better)
    expected_secondary_ranks = rankdata(-enrichment, method="average")
    expected_log_ranks = -1 * np.log10(expected_secondary_ranks) + np.log10(
        np.max(expected_secondary_ranks)
    )

    actual_log_ranks = transform(pvalues, enrichment)
    np.testing.assert_array_almost_equal(actual_log_ranks, expected_log_ranks)


def test_no_ties_in_primary_column():
    pvalues = np.array([0.01, 0.02, 0.03, 0.04])
    enrichment = np.array([5.0, 10.0, 15.0, 20.0])

    # With no ties in pvalue, the secondary column should have no effect
    expected_ranks = rankdata(pvalues, method="average")
    expected_log_ranks = -1 * np.log10(expected_ranks) + np.log10(
        np.max(expected_ranks)
    )

    actual_log_ranks = transform(pvalues, enrichment)
    np.testing.assert_array_almost_equal(actual_log_ranks, expected_log_ranks)


def test_tied_in_both_pvalue_and_enrichment():
    pvalues = np.array([0.01, 0.05, 0.01, 0.02, 0.05])
    enrichment = np.array([5.0, 3.0, 5.0, 4.0, 3.0])

    # With ties in both primary and secondary columns
    expected_ranks = np.array([1.5, 4.5, 1.5, 3.0, 4.5])
    expected_log_ranks = -1 * np.log10(expected_ranks) + np.log10(
        np.max(expected_ranks)
    )

    actual_log_ranks = transform(pvalues, enrichment)
    np.testing.assert_array_almost_equal(actual_log_ranks, expected_log_ranks)
