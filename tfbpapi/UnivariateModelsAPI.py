import asyncio
import json
import os
import time
from typing import Any

import aiohttp
import pandas as pd
import requests  # type: ignore

from yeastdnnexplorer.interface.AbstractRecordsOnlyAPI import AbstractRecordsOnlyAPI


class UnivariateModelsAPI(AbstractRecordsOnlyAPI):
    """
    A class to interact with the UnivariateModels API.

    Retrieves univariatemodels data from the database.

    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the UnivariateModels object. This will serve as an interface to the
        UnivariateModels endpoint of both the database and the application cache.

        :param url: The URL of the UnivariateModels API
        :param kwargs: Additional parameters to pass to AbstractAPI.

        """

        self.bulk_update_url_suffix = kwargs.pop(
            "bulk_update_url_suffix", "bulk-update"
        )

        super().__init__(
            url=kwargs.pop("url", os.getenv("UNIVARIATEMODELS_URL", "")),
            **kwargs,
        )

    async def submit(
        self,
        post_dict: dict[str, Any],
        **kwargs,
    ) -> Any:
        """
        Submit a UnivariateModels task to the UnivariateModels API.

        :param post_dict: The dictionary to submit to the UnivariateModels API. The
            typing needs to be adjusted -- it can take a list of dictionaries to submit
            a batch.
        :return: The group_task_id of the submitted task.

        """
        # make a post request with the post_dict to univariatemodels_url
        univariatemodels_url = f"{self.url.rstrip('/')}/submit/"
        self.logger.debug("univariatemodels_url: %s", univariatemodels_url)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                univariatemodels_url, headers=self.header, json=post_dict
            ) as response:
                try:
                    response.raise_for_status()
                except aiohttp.ClientResponseError as e:
                    self.logger.error(
                        "Failed to submit UnivariateModels task: Status %s, Reason %s",
                        e.status,
                        e.message,
                    )
                    raise
                result = await response.json()
                try:
                    return result["group_task_id"]
                except KeyError:
                    self.logger.error(
                        "Expected 'group_task_id' in response: %s", json.dumps(result)
                    )
                    raise

    async def retrieve(
        self,
        group_task_id: str,
        timeout: int = 300,
        polling_interval: int = 2,
        **kwargs,
    ) -> dict[str, pd.DataFrame]:
        """
        Periodically check the task status and retrieve the result when the task
        completes.

        :param group_task_id: The task ID to retrieve results for.
        :param timeout: The maximum time to wait for the task to complete (in seconds).
        :param polling_interval: The time to wait between status checks (in seconds).
        :return: Records from the UnivariateModels API of the successfully completed
            task.

        """
        # Start time for timeout check
        start_time = time.time()

        # Task status URL
        status_url = f"{self.url.rstrip('/')}/status/"

        while True:
            async with aiohttp.ClientSession() as session:
                # Send a GET request to check the task status
                async with session.get(
                    status_url,
                    headers=self.header,
                    params={"group_task_id": group_task_id},
                ) as response:
                    response.raise_for_status()  # Raise an error for bad status codes
                    status_response = await response.json()

                    # Check if the task is complete
                    if status_response.get("status") == "SUCCESS":

                        if error_tasks := status_response.get("error_tasks"):
                            self.logger.error(
                                f"Tasks {group_task_id} failed: {error_tasks}"
                            )
                        if success_tasks := status_response.get("success_pks"):
                            params = {"id": ",".join(str(pk) for pk in success_tasks)}
                            return await self.read(params=params)
                    elif status_response.get("status") == "FAILURE":
                        raise Exception(
                            f"Task {group_task_id} failed: {status_response}"
                        )

                    # Check if we have reached the timeout
                    elapsed_time = time.time() - start_time
                    if elapsed_time > timeout:
                        raise TimeoutError(
                            f"Task {group_task_id} did not "
                            "complete within {timeout} seconds."
                        )

                    # Wait for the specified polling interval before checking again
                    await asyncio.sleep(polling_interval)

    def create(self, data: dict[str, Any], **kwargs) -> requests.Response:
        raise NotImplementedError("The UnivariateModels does not support create.")

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
        """
        Delete a UnivariateModels record from the database.

        :param id: The ID of the UnivariateModels record to delete.
        :return: A dictionary with a status message indicating success or failure.

        """
        # Include the Authorization header with the token
        headers = kwargs.get("headers", {})
        headers["Authorization"] = f"Token {self.token}"

        # Make the DELETE request with the updated headers
        response = requests.delete(f"{self.url}/{id}/", headers=headers, **kwargs)

        if response.status_code == 204:
            return {
                "status": "success",
                "message": "UnivariateModels deleted successfully.",
            }

        # Raise an error if the response indicates failure
        response.raise_for_status()
