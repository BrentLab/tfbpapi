"""
Simplified rank-response analysis for pre-arranged binding and perturbation data.

This module provides a streamlined approach to rank-response analysis where:
1. Binding data is already ranked by strength (rank 1 = strongest binding)
2. Perturbation data provides a simple responsive TRUE/FALSE column
3. Analysis focuses on binning and statistical calculations

"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class RankResponseAnalyzer:
    """
    Simplified rank-response analyzer for pre-arranged data.

    Takes a DataFrame with target identifiers (pre-sorted by binding strength) and
    responsive boolean values, then performs binning analysis.

    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        responsive_col: str,
        bin_size: int = 100,
    ):
        """
        Initialize the rank-response analyzer.

        :param data: DataFrame with target identifiers and responsive booleans
        :param target_col: Name of column containing target identifiers
        :param responsive_col: Name of column containing TRUE/FALSE responsive values
        :param bin_size: Number of targets per bin for analysis
        :raises ValueError: If data validation fails

        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Store parameters
        self.target_col = target_col
        self.responsive_col = responsive_col
        self.bin_size = bin_size

        # Validate and store data
        self.data = self._validate_data(data)
        self.n_targets = len(self.data)
        self.n_bins = (self.n_targets + bin_size - 1) // bin_size  # Ceiling division

        # Calculate overall statistics
        self.total_responsive = self.data[responsive_col].sum()
        self.overall_response_rate = self.total_responsive / self.n_targets

        self.logger.info(
            f"Initialized RankResponseAnalyzer: {self.n_targets} targets, "
            f"{self.total_responsive} responsive ({self.overall_response_rate:.1%}), "
            f"{self.n_bins} bins of size {bin_size}"
        )

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate input data and return cleaned version."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        if len(data) == 0:
            raise ValueError("Data cannot be empty")

        # Check required columns exist
        if self.target_col not in data.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data")
        if self.responsive_col not in data.columns:
            raise ValueError(
                f"Responsive column '{self.responsive_col}' not found in data"
            )

        # Extract just the required columns
        clean_data = data[[self.target_col, self.responsive_col]].copy()

        # Check for missing values
        if clean_data[self.target_col].isna().any():
            raise ValueError(
                f"Target column '{self.target_col}' contains missing values"
            )
        if clean_data[self.responsive_col].isna().any():
            raise ValueError(
                f"Responsive column '{self.responsive_col}' contains missing values"
            )

        # Validate responsive column is boolean-like
        unique_values = set(clean_data[self.responsive_col].unique())
        valid_boolean_sets = [
            {True, False},
            {1, 0},
            {1.0, 0.0},
            {"TRUE", "FALSE"},
            {"True", "False"},
            {"true", "false"},
            {"T", "F"},
        ]

        # Allow subsets (e.g., only True values, only False values)
        is_valid_boolean = any(
            unique_values.issubset(valid_set) for valid_set in valid_boolean_sets
        )

        if not is_valid_boolean:
            raise ValueError(
                f"Responsive column '{self.responsive_col}' must contain boolean-like values. "
                f"Found: {unique_values}"
            )

        # Convert to standard boolean
        clean_data[self.responsive_col] = clean_data[self.responsive_col].astype(bool)

        # Reset index to ensure proper ranking (rank 1 = index 0)
        clean_data = clean_data.reset_index(drop=True)

        self.logger.debug(
            f"Validated data: {len(clean_data)} rows, {unique_values} -> boolean"
        )

        return clean_data

    def create_bins(self) -> pd.DataFrame:
        """
        Create bins from the ranked data.

        :return: DataFrame with bin assignments for each target

        """
        bins_data = self.data.copy()
        bins_data["rank"] = range(1, len(bins_data) + 1)
        bins_data["bin"] = ((bins_data["rank"] - 1) // self.bin_size) + 1

        return bins_data

    def calculate_bin_stats(self) -> pd.DataFrame:
        """
        Calculate statistics for each bin.

        :return: DataFrame with bin-level statistics

        """
        bins_data = self.create_bins()

        bin_stats = []
        for bin_num in range(1, self.n_bins + 1):
            bin_data = bins_data[bins_data["bin"] == bin_num]

            n_targets_in_bin = len(bin_data)
            n_responsive_in_bin = bin_data[self.responsive_col].sum()
            response_rate = (
                n_responsive_in_bin / n_targets_in_bin if n_targets_in_bin > 0 else 0
            )

            # Calculate rank range for this bin
            min_rank = bin_data["rank"].min()
            max_rank = bin_data["rank"].max()

            bin_stats.append(
                {
                    "bin": bin_num,
                    "min_rank": min_rank,
                    "max_rank": max_rank,
                    "n_targets": n_targets_in_bin,
                    "n_responsive": n_responsive_in_bin,
                    "response_rate": response_rate,
                    "enrichment_vs_overall": (
                        response_rate / self.overall_response_rate
                        if self.overall_response_rate > 0
                        else np.nan
                    ),
                }
            )

        return pd.DataFrame(bin_stats)

    def get_bin_summary(self) -> pd.DataFrame:
        """Get comprehensive bin-level summary statistics."""
        return self.calculate_bin_stats()

    def calculate_enrichment(self, reference_rate: float | None = None) -> pd.DataFrame:
        """
        Calculate enrichment scores for each bin.

        :param reference_rate: Reference response rate for enrichment calculation. If
            None, uses overall response rate.
        :return: DataFrame with enrichment calculations

        """
        if reference_rate is None:
            reference_rate = self.overall_response_rate

        if reference_rate <= 0:
            raise ValueError(
                "Reference rate must be greater than 0 for enrichment calculation"
            )

        bin_stats = self.calculate_bin_stats()
        bin_stats["enrichment_vs_reference"] = (
            bin_stats["response_rate"] / reference_rate
        )
        bin_stats["reference_rate"] = reference_rate

        return bin_stats

    def get_rank_response_curve(self, window_size: int | None = None) -> pd.DataFrame:
        """
        Get data for plotting rank vs response rate curve with sliding window.

        :param window_size: Size of sliding window for smoothing. If None, uses
            bin_size.
        :return: DataFrame with rank positions and smoothed response rates

        """
        if window_size is None:
            window_size = self.bin_size

        bins_data = self.create_bins()
        curve_data = []

        for i in range(len(bins_data)):
            # Define window around current position
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(bins_data), i + window_size // 2 + 1)

            window_data = bins_data.iloc[start_idx:end_idx]
            window_response_rate = window_data[self.responsive_col].mean()

            curve_data.append(
                {
                    "rank": bins_data.iloc[i]["rank"],
                    "target": bins_data.iloc[i][self.target_col],
                    "responsive": bins_data.iloc[i][self.responsive_col],
                    "smoothed_response_rate": window_response_rate,
                    "window_size": len(window_data),
                }
            )

        return pd.DataFrame(curve_data)

    def get_summary_stats(self) -> dict[str, Any]:
        """Get overall summary statistics."""
        return {
            "n_targets": self.n_targets,
            "n_responsive": self.total_responsive,
            "overall_response_rate": self.overall_response_rate,
            "n_bins": self.n_bins,
            "bin_size": self.bin_size,
            "targets_per_bin_avg": self.n_targets / self.n_bins,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Export full results as a comprehensive DataFrame."""
        bins_data = self.create_bins()
        bin_stats = self.calculate_bin_stats()

        # Merge bin statistics back to individual target data
        result = bins_data.merge(bin_stats, on="bin", suffixes=("", "_bin"))

        return result

    def __repr__(self) -> str:
        """String representation of the analyzer."""
        return (
            f"RankResponseAnalyzer("
            f"targets={self.n_targets}, "
            f"responsive={self.total_responsive}, "
            f"rate={self.overall_response_rate:.1%}, "
            f"bins={self.n_bins})"
        )
