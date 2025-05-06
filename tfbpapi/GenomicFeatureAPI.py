import os
from typing import Any

import pandas as pd

from tfbpapi.AbstractRecordsOnlyAPI import AbstractRecordsOnlyAPI


class GenomicFeatureAPI(AbstractRecordsOnlyAPI):
    """A class to interact with the GenomicFeatureAPI endpoint."""

    def __init__(self, **kwargs):
        """
        Initialize the GenomicFeatureAPI object.

        :param kwargs: parameters to pass to AbstractAPI via AbstractRecordsOnlyAPI.

        """
        valid_param_keys = kwargs.pop(
            "valid_param_keys",
            [
                "id",
                "chr",
                "start",
                "end",
                "strand",
                "type",
                "locus_tag",
                "symbol",
                "source",
                "alias",
                "note",
            ],
        )

        url = kwargs.pop("url", os.getenv("GENOMICFEATURE_URL", None))
        if not url:
            raise AttributeError(
                "url must be provided or the environmental variable ",
                "`GENOMICFEATURE_URL` must be set",
            )

        super().__init__(url=url, valid_keys=valid_param_keys, **kwargs)

    def create(self, data: dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("The GenomicFeatureAPI does not support create.")

    def update(self, df: pd.DataFrame, **kwargs) -> Any:
        raise NotImplementedError("The GenomicFeatureAPI does not support update.")

    def delete(self, id: str, **kwargs) -> Any:
        raise NotImplementedError("The GenomicFeatureAPI does not support delete.")

    def submit(self, post_dict: dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("The GenomicFeatureAPI does not support submit.")

    def retrieve(
        self, group_task_id: str, timeout: int, polling_interval: int, **kwargs
    ) -> Any:
        raise NotImplementedError("The GenomicFeatureAPI does not support retrieve.")
