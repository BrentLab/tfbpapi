import numpy as np
from scipy.stats import rankdata


def shifted_negative_log_ranks(ranks: np.ndarray) -> np.ndarray:
    """
    Transforms ranks to negative log10 values and shifts such that the lowest value is
    0.

    :param ranks: A vector of ranks
    :return np.ndarray: A vector of negative log10 transformed ranks shifted such that
        the lowest value is 0
    :raises ValueError: If the ranks are not numeric.

    """
    if not np.issubdtype(ranks.dtype, np.number):
        raise ValueError("`ranks` must be a numeric")
    max_rank = np.max(ranks)
    log_max_rank = np.log10(max_rank)
    return -1 * np.log10(ranks) + log_max_rank


def stable_rank(
    pvalue_vector: np.ndarray, enrichment_vector: np.ndarray, method="average"
) -> np.ndarray:
    """
    Ranks data by primary_column, breaking ties based on secondary_column. The expected
    primary and secondary columns are 'pvalue' and 'enrichment', respectively. Then the
    ranks are transformed to negative log10 values and shifted such that the lowest
    value is 0 and the highest value is log10(min_rank).

    :param pvalue_vector: A vector of pvalues
    :param enrichment_vector: A vector of enrichment values corresponding to the pvalues
    :param method: The method to use for final ranking. Default is "average".
        See `rankdata`

    :return np.ndarray: A vector of negative log10 transformed ranks shifted such that
        the lowest value is 0 and the highest value is log10(min_rank)
    :raises ValueError: If the primary or secondary column is not numeric.

    """

    # Check if primary and secondary columns are numeric
    if not np.issubdtype(pvalue_vector.dtype, np.number):
        raise ValueError("`primary_vector` must be a numeric")
    if not np.issubdtype(enrichment_vector.dtype, np.number):
        raise ValueError("`secondary_vector` must be a numeric")

    # Step 1: Rank by primary_column
    # note that this will now always be an integer, unlike average which could return
    # decimal values making adding the secondary rank more difficult
    primary_rank = rankdata(pvalue_vector, method="min")

    # Step 2: Identify ties in primary_rank
    unique_ranks = np.unique(primary_rank)

    # Step 3: Adjust ranks within ties using secondary ranking
    adjusted_primary_rank = primary_rank.astype(
        float
    )  # Convert to float for adjustments

    for unique_rank in unique_ranks:
        # Get indices where primary_rank == unique_rank
        tie_indices = np.where(primary_rank == unique_rank)[0]

        if len(tie_indices) > 1:  # Only adjust if there are ties
            # Rank within the tie group by secondary_column
            # (descending if higher is better)
            tie_secondary_values = enrichment_vector[tie_indices]
            secondary_rank_within_ties = rankdata(
                -tie_secondary_values, method="average"
            )

            # Calculate dynamic scale factor to ensure adjustments are < 1. Since the
            # primary_rank is an integer, adding a number less than 1 will not affect
            # rank relative to the other groups.
            max_secondary_rank = np.max(secondary_rank_within_ties)
            scale_factor = (
                0.9 / max_secondary_rank
            )  # Keep scale factor slightly below 1/max rank

            # multiple the secondary_rank_within_ties values by 0.1 and add this value
            # to the adjusted_primary_rank_values. This will rank the tied primary
            # values by the secondary values, but not affect the overall primary rank
            # outside of the tie group
            # think about this scale factor
            adjusted_primary_rank[tie_indices] += (
                secondary_rank_within_ties * scale_factor
            )

    # Step 4: Final rank based on the adjusted primary ranks
    final_ranks = rankdata(adjusted_primary_rank, method=method)

    return final_ranks


def rank_by_pvalue(pvalue_vector: np.ndarray, method="average") -> np.ndarray:
    """
    This expects a vector of pvalues, returns a vector of ranks where the lowest pvalue
    has the lowest rank.

    :param pvalue_vector: A vector of pvalues
    :param enrichment_vector: A vector of enrichment values corresponding to the pvalues
    :param method: The method to use for ranking. Default is "average". See `rankdata`
    :return np.ndarray: A vector of negative log10 transformed ranks shifted such that
        the lowest value is 0 and the highest value is log10(min_rank)
    :raises ValueError: If the primary or secondary column is not numeric.

    """

    # Check if primary and secondary columns are numeric
    if not np.issubdtype(pvalue_vector.dtype, np.number):
        raise ValueError("`primary_vector` must be a numeric")

    # Step 1: Rank by primary_column
    # note that this will now always be an integer, unlike average which could return
    # decimal values making adding the secondary rank more difficult
    return rankdata(pvalue_vector, method=method)


def transform(
    pvalue_vector: np.ndarray,
    enrichment_vector: np.ndarray,
    use_enrichment: bool = True,
    negative_log_shift: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    This calls the rank() function and then transforms the ranks to negative log10
    values and shifts to the right such that the lowest value (largest rank, least
    important) is 0.

    :param pvalue_vector: A vector of pvalues
    :param enrichment_vector: A vector of enrichment values corresponding to the pvalues
    :param use_enrichment: Set to True to use the enrichment vector to break ties.
        Default is True. If False, pvalues will be ranked directly with method="average'
    :param negative_log_shift: Set to True to shift the ranks to the right such that the
        lowest value (largest rank, least important) is 0. Default is True.
    :param kwargs: Additional keyword arguments to pass to the rank() function (e.g.
        method="min")
    :return np.ndarray: A vector of negative log10 transformed ranks shifted such that
        the lowest value is 0 and the highest value is log10(min_rank)
    :raises ValueError: If the primary or secondary column is not numeric.

    """
    if use_enrichment:
        ranks = stable_rank(pvalue_vector, enrichment_vector, **kwargs)
    else:
        ranks = rank_by_pvalue(pvalue_vector, **kwargs)

    if negative_log_shift:
        return shifted_negative_log_ranks(ranks)
    else:
        return ranks
