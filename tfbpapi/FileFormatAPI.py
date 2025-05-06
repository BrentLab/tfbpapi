import os
from typing import Any

import pandas as pd

from tfbpapi.AbstractRecordsOnlyAPI import AbstractRecordsOnlyAPI


class FileFormatAPI(AbstractRecordsOnlyAPI):
    """A class to interact with the FileFormatAPI endpoint."""

    def __init__(self, **kwargs):
        """
        Initialize the FileFormatAPI object.

        :param kwargs: parameters to pass to AbstractAPI via AbstractRecordsOnlyAPI.

        """
        valid_param_keys = kwargs.pop(
            "valid_param_keys",
            [
                "fileformat",
                "fields",
                "separator",
                "feature_identifier_col",
                "effect_col",
                "default_effect_threshold",
                "pval_col",
                "default_pvalue_threshold",
            ],
        )

        url = kwargs.pop("url", os.getenv("FILEFORMAT_URL", None))
        if not url:
            raise AttributeError(
                "url must be provided or the environmental variable ",
                "`FILEFORMAT_URL` must be set",
            )

        super().__init__(url=url, valid_keys=valid_param_keys, **kwargs)

    def create(self, data: dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("The FileFormatAPI does not support create.")

    def update(self, df: pd.DataFrame, **kwargs) -> Any:
        raise NotImplementedError("The FileFormatAPI does not support update.")

    def delete(self, id: str, **kwargs) -> Any:
        raise NotImplementedError("The FileFormatAPI does not support delete.")

    def submit(self, post_dict: dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("The FileFormatAPI does not support submit.")

    def retrieve(
        self, group_task_id: str, timeout: int, polling_interval: int, **kwargs
    ) -> Any:
        raise NotImplementedError("The FileFormatAPI does not support retrieve.")
