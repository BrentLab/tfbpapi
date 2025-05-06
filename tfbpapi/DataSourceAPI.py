import os
from typing import Any

import pandas as pd

from tfbpapi.AbstractRecordsOnlyAPI import AbstractRecordsOnlyAPI


class DataSourceAPI(AbstractRecordsOnlyAPI):
    """A class to interact with the DataSourceAPI endpoint."""

    def __init__(self, **kwargs):
        """
        Initialize the DataSourceAPI object.

        :param kwargs: parameters to pass to AbstractAPI via AbstractRecordsOnlyAPI.

        """
        valid_param_keys = kwargs.pop(
            "valid_param_keys",
            ["id", "fileformat_id", "fileformat", "lab", "assay", "workflow"],
        )

        url = kwargs.pop("url", os.getenv("DATASOURCE_URL", None))
        if not url:
            raise AttributeError(
                "url must be provided or the environmental variable ",
                "`DATASOURCE_URL` must be set",
            )

        super().__init__(url=url, valid_keys=valid_param_keys, **kwargs)

    def create(self, data: dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("The DataSourceAPI does not support create.")

    def update(self, df: pd.DataFrame, **kwargs) -> Any:
        raise NotImplementedError("The DataSourceAPI does not support update.")

    def delete(self, id: str, **kwargs) -> Any:
        raise NotImplementedError("The DataSourceAPI does not support delete.")

    def submit(self, post_dict: dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("The DataSourceAPI does not support submit.")

    def retrieve(
        self, group_task_id: str, timeout: int, polling_interval: int, **kwargs
    ) -> Any:
        raise NotImplementedError("The DataSourceAPI does not support retrieve.")
