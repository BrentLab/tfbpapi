import os
from typing import Any

import pandas as pd
import requests  # type: ignore

from tfbpapi.AbstractRecordsOnlyAPI import AbstractRecordsOnlyAPI


class ExpressionManualQCAPI(AbstractRecordsOnlyAPI):
    """A class to interact with the ExpressionManualQCAPI endpoint."""

    def __init__(self, **kwargs):
        """
        Initialize the ExpressionManualQCAPI object.

        :param kwargs: parameters to pass to AbstractAPI via AbstractRecordsOnlyAPI.

        """
        valid_param_keys = kwargs.pop(
            "valid_param_keys",
            [
                "id",
                "expression",
                "strain_verified",
                "regulator_locus_tag",
                "regulator_symbol",
                "batch",
                "replicate",
                "control",
                "mechanism",
                "restriction",
                "time",
                "source",
                "lab",
                "assay",
                "workflow",
            ],
        )

        url = kwargs.pop("url", os.getenv("EXPRESSIONMANUALQC_URL", None))
        if not url:
            raise AttributeError(
                "url must be provided or the environmental variable ",
                "`EXPRESSIONMANUALQC_URL` must be set",
            )

        self.bulk_update_url_suffix = kwargs.pop(
            "bulk_update_url_suffix", "bulk-update"
        )

        super().__init__(url=url, valid_keys=valid_param_keys, **kwargs)

    def create(self, data: dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("The ExpressionManualQCAPI does not support create.")

    def update(self, df: pd.DataFrame, **kwargs: Any) -> requests.Response:
        """
        Update the records in the database.

        :param df: The DataFrame containing the records to update.
        :type df: pd.DataFrame
        :param kwargs: Additional fields to include in the payload.
        :type kwargs: Any
        :return: The response from the POST request.
        :rtype: requests.Response
        :raises requests.RequestException: If the request fails.

        """
        bulk_update_url = (
            f"{self.url.rstrip('/')}/{self.bulk_update_url_suffix.rstrip('/')}/"
        )

        self.logger.debug("bulk_update_url: %s", bulk_update_url)

        # Include additional fields in the payload if provided
        payload = {"data": df.to_dict(orient="records")}
        payload.update(kwargs)

        try:
            response = requests.post(
                bulk_update_url,
                headers=self.header,
                json=payload,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            self.logger.error(f"Error in POST request: {e}")
            raise

    def delete(self, id: str, **kwargs) -> Any:
        raise NotImplementedError("The ExpressionManualQCAPI does not support delete.")

    def submit(self, post_dict: dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("The ExpressionManualQCAPI does not support submit.")

    def retrieve(
        self, group_task_id: str, timeout: int, polling_interval: int, **kwargs
    ) -> Any:
        raise NotImplementedError(
            "The ExpressionManualQCAPI does not support retrieve."
        )
