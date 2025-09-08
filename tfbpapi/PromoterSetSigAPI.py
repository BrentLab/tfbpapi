import os
from typing import Any

import pandas as pd

from tfbpapi.AbstractRecordsAndFilesAPI import (
    AbstractRecordsAndFilesAPI,
)


class PromoterSetSigAPI(AbstractRecordsAndFilesAPI):
    """Class to interact with the PromoterSetSigAPI endpoint."""

    def __init__(self, **kwargs) -> None:
        """
        Initialize the PromoterSetSigAPI object.

        :param kwargs: parameters to pass through AbstractRecordsAndFilesAPI to
            AbstractAPI.

        """
        valid_param_keys = kwargs.pop(
            "valid_param_keys",
            [
                "id",
                "single_binding",
                "composite_binding",
                "promoter",
                "promoter_name",
                "background",
                "background_name",
                "regulator_locus_tag",
                "regulator_symbol",
                "batch",
                "replicate",
                "source",
                "source_name",
                "lab",
                "assay",
                "workflow",
                "data_usable",
                "aggregated",
                "condition",
                "deduplicate",
                "preferred_replicate",
            ],
        )

        url = kwargs.pop("url", os.getenv("PROMOTERSETSIG_URL", None))

        super().__init__(url=url, valid_keys=valid_param_keys, **kwargs)

    def create(self, data: dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("The PromoterSetSigAPI does not support create.")

    def update(self, df: pd.DataFrame, **kwargs) -> Any:
        raise NotImplementedError("The PromoterSetSigAPI does not support update.")

    def delete(self, id: str, **kwargs) -> Any:
        raise NotImplementedError("The PromoterSetSigAPI does not support delete.")

    def submit(self, post_dict: dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("The PromoterSetSigAPI does not support submit.")

    def retrieve(
        self, group_task_id: str, timeout: int, polling_interval: int, **kwargs
    ) -> Any:
        raise NotImplementedError("The PromoterSetSigAPI does not support retrieve.")
