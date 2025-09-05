from __future__ import annotations

from typing import Any, Iterable, Mapping

from .AbstractHfAPI import AbstractHfAPI


class HfHu2007Reimand2010API(AbstractHfAPI):
    """HF-backed API for BrentLab/hu_2007_reimand_2010 dataset.

    Exposes a minimal interface to scan the Parquet file(s) and return an
    Arrow table along with lightweight repo metadata via the base class.

    Parameters accepted in params for filtering:
    - regulator_locus_tag: Optional[str|Iterable[str]] — equality/IN filter
    - regulator_symbol: Optional[str|Iterable[str]] — equality/IN filter
    - target_locus_tag: Optional[str|Iterable[str]] — equality/IN filter
    - target_symbol: Optional[str|Iterable[str]] — equality/IN filter
    - effect_min/effect_max: Optional[float] — numeric range on log2 fold-change
    - pval_min/pval_max: Optional[float] — numeric range on p-value
    """

    DEFAULT_REPO_ID = "BrentLab/hu_2007_reimand_2010"

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        repo_type: str | None = "dataset",
        revision: str | None = None,
        token: str = "",
        cache_dir: str | None = None,
        local_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_dir=local_dir,
            **kwargs,
        )

    def build_query(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Construct a query plan for this dataset.

        The dataset contains a single Parquet file "hu_2007_reimand_2010.parquet".
        We support basic equality/IN and numeric range filtering on provided columns.
        """
        # Files to fetch
        file_patterns: list[str] = ["hu_2007_reimand_2010.parquet"]

        filter_expr = None
        filters: list[Any] = []

        # Attempt to construct pushdown expressions; fall back silently if pyarrow not present
        try:
            import pyarrow.dataset as ds  # type: ignore
        except Exception:
            ds = None  # type: ignore

        def as_iterable(value: Any) -> list[Any]:
            if isinstance(value, (list, tuple, set)):
                return list(value)
            return [value]

        if ds is not None:
            # String equality/IN filters
            for col in (
                "regulator_locus_tag",
                "regulator_symbol",
                "target_locus_tag",
                "target_symbol",
            ):
                if col in params and params[col] is not None:
                    values = as_iterable(params[col])
                    if len(values) == 1:
                        filters.append(ds.field(col) == values[0])
                    else:
                        filters.append(ds.field(col).isin(values))

            # Numeric ranges
            if params.get("effect_min") is not None:
                try:
                    filters.append(ds.field("effect") >= float(params["effect_min"]))
                except Exception:
                    pass
            if params.get("effect_max") is not None:
                try:
                    filters.append(ds.field("effect") <= float(params["effect_max"]))
                except Exception:
                    pass
            if params.get("pval_min") is not None:
                try:
                    filters.append(ds.field("pval") >= float(params["pval_min"]))
                except Exception:
                    pass
            if params.get("pval_max") is not None:
                try:
                    filters.append(ds.field("pval") <= float(params["pval_max"]))
                except Exception:
                    pass

            # Combine filters
            valid_filters = [f for f in filters if f is not None]
            if len(valid_filters) == 1:
                filter_expr = valid_filters[0]
            elif len(valid_filters) > 1:
                expr = valid_filters[0]
                for f in valid_filters[1:]:
                    expr = expr & f
                filter_expr = expr

        return {
            "file_patterns": file_patterns,
            "filter": filter_expr,
            "format": "parquet",
        }

    def read_table(self, **kwargs: Any):
        """Convenience: return only the Arrow table.

        Equivalent to `self.read(...)["data"]`.
        """
        result = self.read(params=kwargs.pop("params", {}), **kwargs)
        return result["data"] 