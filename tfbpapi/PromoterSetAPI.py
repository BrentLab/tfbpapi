import os
from typing import Any

import pandas as pd

from tfbpapi.AbstractRecordsAndFilesAPI import (
    AbstractRecordsAndFilesAPI,
)


class PromoterSetAPI(AbstractRecordsAndFilesAPI):
    """Class to interact with the PromoterSetAPI endpoint."""

    def __init__(self, **kwargs) -> None:
        """
        Initialize the PromoterSetAPI object.

        :param kwargs: parameters to pass through AbstractRecordsAndFilesAPI to
            AbstractAPI.

        """
        valid_param_keys = kwargs.pop(
            "valid_param_keys",
            ["id", "name"],
        )

        url = kwargs.pop("url", os.getenv("PROMOTERSET_URL", None))

        super().__init__(url=url, valid_keys=valid_param_keys, **kwargs)

    def create(self, data: dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("The PromoterSetAPI does not support create.")

    def update(self, df: pd.DataFrame, **kwargs) -> Any:
        raise NotImplementedError("The PromoterSetAPI does not support update.")

    def delete(self, id: str, **kwargs) -> Any:
        raise NotImplementedError("The PromoterSetAPI does not support delete.")

    def submit(self, post_dict: dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("The PromoterSetAPI does not support submit.")

    def retrieve(
        self, group_task_id: str, timeout: int, polling_interval: int, **kwargs
    ) -> Any:
        raise NotImplementedError("The PromoterSetAPI does not support retrieve.")
