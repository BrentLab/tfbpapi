import os
from typing import Any

import pandas as pd

from tfbpapi.AbstractRecordsAndFilesAPI import (
    AbstractRecordsAndFilesAPI,
)


class BindingConcatenatedAPI(AbstractRecordsAndFilesAPI):
    """Class to interact with the BindingConcatenatedAPI endpoint."""

    def __init__(self, **kwargs) -> None:
        """
        Initialize the BindingConcatenatedAPI object.

        :param kwargs: parameters to pass through AbstractRecordsAndFilesAPI to
            AbstractAPI.

        """
        valid_param_keys = kwargs.pop(
            "valid_param_keys",
            [
                "id",
                "regulator",
                "regulator_locus_tag",
                "regulator_symbol",
                "batch",
                "replicate",
                "source",
                "strain",
                "condition",
                "lab",
                "assay",
                "workflow",
                "data_usable",
            ],
        )

        url = kwargs.pop("url", os.getenv("BINDINGCONCATENATED_URL", None))

        super().__init__(url=url, valid_keys=valid_param_keys, **kwargs)

    def create(self, data: dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("The BindingConcatenatedAPI does not support create.")

    def update(self, df: pd.DataFrame, **kwargs) -> Any:
        raise NotImplementedError("The BindingConcatenatedAPI does not support update.")

    def delete(self, id: str, **kwargs) -> Any:
        raise NotImplementedError("The BindingConcatenatedAPI does not support delete.")

    def submit(self, post_dict: dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("The BindingConcatenatedAPI does not support submit.")

    def retrieve(
        self, group_task_id: str, timeout: int, polling_interval: int, **kwargs
    ) -> Any:
        raise NotImplementedError(
            "The BindingConcatenatedAPI does not support retrieve."
        )
