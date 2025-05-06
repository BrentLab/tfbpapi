import os
from typing import Any

import pandas as pd

from tfbpapi.AbstractRecordsOnlyAPI import AbstractRecordsOnlyAPI


class RegulatorAPI(AbstractRecordsOnlyAPI):
    """A class to interact with the RegulatorAPI endpoint."""

    def __init__(self, **kwargs):
        """
        Initialize the RegulatorAPI object.

        :param kwargs: parameters to pass to AbstractAPI via AbstractRecordsOnlyAPI.

        """
        valid_param_keys = kwargs.pop(
            "valid_param_keys",
            [
                "id",
                "regulator_locus_tag",
                "regulator_symbol",
                "under_development",
            ],
        )

        url = kwargs.pop("url", os.getenv("REGULATOR_URL", None))
        if not url:
            raise AttributeError(
                "url must be provided or the environmental variable ",
                "`REGULATOR_URL` must be set",
            )

        super().__init__(url=url, valid_keys=valid_param_keys, **kwargs)

    def create(self, data: dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("The RegulatorAPI does not support create.")

    def update(self, df: pd.DataFrame, **kwargs) -> Any:
        raise NotImplementedError("The RegulatorAPI does not support update.")

    def delete(self, id: str, **kwargs) -> Any:
        raise NotImplementedError("The RegulatorAPI does not support delete.")

    def submit(self, post_dict: dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("The RegulatorAPI does not support submit.")

    def retrieve(
        self, group_task_id: str, timeout: int, polling_interval: int, **kwargs
    ) -> Any:
        raise NotImplementedError("The RegulatorAPI does not support retrieve.")
