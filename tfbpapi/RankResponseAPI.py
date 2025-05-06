import asyncio
import json
import os
import tarfile
import tempfile
import time
from typing import Any

import aiohttp
import pandas as pd
from requests import Response, delete, post  # type: ignore
from requests_toolbelt import MultipartEncoder

from tfbpapi.AbstractRecordsAndFilesAPI import (
    AbstractRecordsAndFilesAPI,
)


class RankResponseAPI(AbstractRecordsAndFilesAPI):
    """
    A class to interact with the Rank Response API.

    Retrieves rank response data from the database.

    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the RankResponseAPI object. This will serve as an interface to the
        RankResponse endpoint of both the database and the application cache.

        :param url: The URL of the Rank Response API
        :param kwargs: Additional parameters to pass to AbstractAPI.

        """
        super().__init__(
            url=kwargs.pop("url", os.getenv("RANKRESPONSE_URL", "")),
            **kwargs,
        )

    async def submit(
        self,
        post_dict: dict[str, Any],
        **kwargs,
    ) -> Any:
        # make a post request with the post_dict to rankresponse_url
        rankresponse_url = f"{self.url.rstrip('/')}/submit/"
        self.logger.debug("rankresponse_url: %s", rankresponse_url)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                rankresponse_url, headers=self.header, json=post_dict
            ) as response:
                response.raise_for_status()
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
        :return: Extracted files from the result tarball.

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
                        # Fetch and return the tarball
                        return await self._download_result(group_task_id)
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

    async def _download_result(self, group_task_id: str) -> Any:
        """
        Download the result tarball after the task is successful.

        :param group_task_id: The group_task_id to download the results for.
        :return: Extracted metadata and data from the tarball.

        """
        download_url = f"{self.url.rstrip('/')}/retrieve_task/"

        async with aiohttp.ClientSession() as session:
            async with session.get(
                download_url,
                headers=self.header,
                params={"group_task_id": group_task_id},
            ) as response:
                response.raise_for_status()  # Ensure request was successful
                tar_data = await response.read()

                # Save tarball to a temporary file or return raw tar content
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".tar.gz"
                ) as temp_file:
                    temp_file.write(tar_data)
                    temp_file.flush()
                    temp_file.seek(0)

                    # Extract and return the content of the tarball
                    return self._extract_files(temp_file.name)

    def _extract_files(self, tar_path: str) -> dict[str, pd.DataFrame]:
        """
        Extract metadata and associated files from a tarball.

        :param tar_path: The path to the tarball file.
        :return: A tuple of metadata DataFrame and a dictionary of DataFrames for each
            file.

        """
        with tarfile.open(tar_path, mode="r:gz") as tar:
            tar_members = tar.getmembers()

            # Extract metadata.json
            metadata_member = next(
                (m for m in tar_members if m.name == "metadata.json"), None
            )
            if metadata_member is None:
                raise FileNotFoundError("metadata.json not found in tar archive")

            extracted_file = tar.extractfile(metadata_member)
            if extracted_file is None:
                raise FileNotFoundError("Failed to extract metadata.json")

            with extracted_file as f:
                metadata_dict = json.load(f)

            metadata_df = pd.DataFrame(metadata_dict.values())
            metadata_df["id"] = metadata_dict.keys()

            # Extract CSV files
            data = {}
            for rr_id in metadata_df["id"]:
                csv_filename = f"{rr_id}.csv.gz"
                member = next((m for m in tar_members if m.name == csv_filename), None)
                if member is None:
                    raise FileNotFoundError(f"{csv_filename} not found in tar archive")

                extracted_file = tar.extractfile(member)
                if extracted_file is None:
                    raise FileNotFoundError(f"Failed to extract {csv_filename}")

                with extracted_file as f:
                    data[rr_id] = pd.read_csv(f, compression="gzip")
        return {"metadata": metadata_df, "data": data}

    def create(self, data: dict[str, Any], **kwargs) -> Response:
        """
        Create a new RankResponse record by uploading a gzipped CSV file.

        :param data: This should be the fields in the RankREsponse model, eg
            "promotersetsig_id", "expression_id" and "parameters".
        :param kwargs: Additional parameters to pass to the post. This must include a
            DataFrame to upload as a CSV file with the keyword `df`, eg `df=my_df`.

        :return: The result of the post request.

        :raises ValueError: If a DataFrame is not provided in the keyword arguments.
        :raises TypeError: If the DataFrame provided is not a pandas DataFrame.

        """
        # ensure that the url ends in a slash
        rankresponse_url = f"{self.url.rstrip('/')}/"
        df = kwargs.pop("df", None)

        if df is None:
            raise ValueError(
                "A DataFrame must be provided to create "
                "a RankResponse via keyword `df`"
            )
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected a DataFrame for keyword `df`, got {type(df).__name__}"
            )

        # Create a temporary gzipped CSV file from the DataFrame
        with tempfile.NamedTemporaryFile(suffix=".csv.gz") as temp_file:
            df.to_csv(temp_file.name, compression="gzip", index=False)

            # Prepare the file and metadata for upload
            with open(temp_file.name, "rb") as file:
                multipart_data = MultipartEncoder(
                    fields={**data, "file": (temp_file.name, file, "application/gzip")}
                )
                headers = {**self.header, "Content-Type": multipart_data.content_type}

                # Send the POST request with custom encoded multipart data
                response = post(rankresponse_url, headers=headers, data=multipart_data)

        response.raise_for_status()
        return response

    def update(self, df: pd.DataFrame, **kwargs) -> Any:
        raise NotImplementedError("The RankResponseAPI does not support update.")

    def delete(self, id: str = "", **kwargs) -> Any:
        """
        Delete one or more records from the database.

        :param id: The ID of the record to delete. However, you can also pass in
        `ids` as a list of IDs to delete multiple records. This is why `id` is optional.
        If neither `id` nor `ids` is provided, a ValueError is raised.

        :return: A dictionary with a status message indicating success or failure.

        :raises ValueError: If neither `id` nor `ids` is provided.

        """
        # Include the Authorization header with the token
        headers = kwargs.get("headers", {})
        headers["Authorization"] = f"Token {self.token}"

        ids = kwargs.pop("ids", str(id))

        # Determine if it's a single ID or multiple
        if isinstance(ids, str) and str != "":
            # Single ID deletion for backward compatibility
            response = delete(f"{self.url}/{ids}/", headers=headers, **kwargs)
        elif isinstance(ids, list) and ids:
            # Bulk delete with a list of IDs
            response = delete(
                f"{self.url}/delete/",
                headers=headers,
                json={"ids": ids},  # Send the list of IDs in the request body
                **kwargs,
            )
        else:
            raise ValueError(
                "No ID(s) provided for deletion. Either pass a single ID with "
                "`id` or a list of IDs with `ids = [1,2, ...]"
            )

        if response.status_code in [200, 204]:
            return {
                "status": "success",
                "message": "RankResponse(s) deleted successfully.",
            }

        # Raise an error if the response indicates failure
        response.raise_for_status()
