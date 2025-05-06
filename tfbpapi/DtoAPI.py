import asyncio
import json
import os
import time
from typing import Any

import aiohttp
import pandas as pd
import requests  # type: ignore

from tfbpapi.AbstractRecordsOnlyAPI import AbstractRecordsOnlyAPI


class DtoAPI(AbstractRecordsOnlyAPI):
    """
    A class to interact with the DTO API.

    Retrieves dto data from the database.

    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the DTO object. This will serve as an interface to the DTO endpoint
        of both the database and the application cache.

        :param url: The URL of the DTO API
        :param kwargs: Additional parameters to pass to AbstractAPI.

        """

        self.bulk_update_url_suffix = kwargs.pop(
            "bulk_update_url_suffix", "bulk-update"
        )

        super().__init__(
            url=kwargs.pop("url", os.getenv("DTO_URL", "")),
            **kwargs,
        )

    async def read(self, *args, **kwargs) -> Any:
        """
        Override the read() method to use a custom callback that parses metadata.

        :param callback: The function to call with the metadata. Defaults to parsing
            metadata.
        :type callback: Callable[[pd.DataFrame, dict[str, Any] | None, Any], Any]
        :return: The result of the callback function.
        :rtype: Any

        """

        # Define the default callback
        def dto_callback(metadata, data, cache, **kwargs):
            return {"metadata": self.parse_metadata(metadata), "data": data}

        # Explicitly set the callback argument to dto_callback
        kwargs["callback"] = dto_callback

        # Call the superclass method with updated kwargs
        return await super().read(*args, **kwargs)

    async def submit(
        self,
        post_dict: dict[str, Any],
        **kwargs,
    ) -> Any:
        """
        Submit a DTO task to the DTO API.

        :param post_dict: The dictionary to submit to the DTO API. The typing needs to
            be adjusted -- it can take a list of dictionaries to submit a batch.
        :return: The group_task_id of the submitted task.

        """
        # make a post request with the post_dict to dto_url
        dto_url = f"{self.url.rstrip('/')}/submit/"
        self.logger.debug("dto_url: %s", dto_url)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                dto_url, headers=self.header, json=post_dict
            ) as response:
                try:
                    response.raise_for_status()
                except aiohttp.ClientResponseError as e:
                    self.logger.error(
                        "Failed to submit DTO task: Status %s, Reason %s",
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
        :return: Records from the DTO API of the successfully completed task.

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
        raise NotImplementedError("The DTO does not support create.")

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
        Delete a DTO record from the database.

        :param id: The ID of the DTO record to delete.
        :return: A dictionary with a status message indicating success or failure.

        """
        # Include the Authorization header with the token
        headers = kwargs.get("headers", {})
        headers["Authorization"] = f"Token {self.token}"

        # Make the DELETE request with the updated headers
        response = requests.delete(f"{self.url}/{id}/", headers=headers, **kwargs)

        if response.status_code == 204:
            return {"status": "success", "message": "DTO deleted successfully."}

        # Raise an error if the response indicates failure
        response.raise_for_status()

    def parse_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Parse the metadata from the DTO API.

        :param metadata: The metadata DataFrame to parse.
        :return: The parsed metadata DataFrame.
        :raises KeyError: If the metadata DataFrame is missing required columns.

        """
        if metadata.empty:
            self.logger.warning("Metadata is empty")
            return metadata

        output_columns = [
            "id",
            "promotersetsig",
            "expression",
            "regulator_symbol",
            "binding_source",
            "expression_source",
            "passing_fdr",
            "passing_pvalue",
        ]

        # required columns are "result" and output_columns
        missing_req_columns = [
            col for col in ["result"] + output_columns if col not in metadata.columns
        ]
        if missing_req_columns:
            raise KeyError(
                "Metadata is missing required columns: "
                "{', '.join(missing_req_columns)}"
            )

        dto_results_list = []

        # Check and rename keys, logging a warning if a key is missing
        keys_to_rename = {
            "rank1": "binding_rank_threshold",
            "rank2": "perturbation_rank_threshold",
            "set1_len": "binding_set_size",
            "set2_len": "perturbation_set_size",
        }

        for _, row in metadata.iterrows():
            dto_results = json.loads(row.result.replace("'", '"'))

            for old_key, new_key in keys_to_rename.items():
                if old_key in dto_results:
                    dto_results[new_key] = dto_results.pop(old_key)
                else:
                    self.logger.warning(
                        f"Key '{old_key}' missing in row with id '{row.id}'."
                    )

            dto_results["id"] = row.id
            dto_results["promotersetsig"] = row.promotersetsig
            dto_results["expression"] = row.expression
            dto_results["regulator_symbol"] = row.regulator_symbol
            dto_results["binding_source"] = row.binding_source
            dto_results["expression_source"] = row.expression_source
            dto_results["passing_fdr"] = row.passing_fdr
            dto_results["passing_pvalue"] = row.passing_pvalue

            dto_results_list.append(dto_results)

        # Create DataFrame
        result_df = pd.DataFrame(dto_results_list)

        # Reorder columns: output_columns first, followed by others
        reordered_columns = output_columns + [
            col for col in result_df.columns if col not in output_columns
        ]

        return result_df.loc[:, reordered_columns]
