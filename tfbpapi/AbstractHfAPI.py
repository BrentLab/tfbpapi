import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import requests
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from requests import HTTPError

# Constants
MB_TO_BYTES = 1024 * 1024


class RepoTooLargeError(ValueError):
    """Raised when repository exceeds auto-download threshold."""


class AbstractHfAPI(ABC):
    """Abstract base class for creating Hugging Face API clients."""

    def __init__(
        self,
        repo_id: str,
        repo_type: Literal["model", "dataset", "space"] = "dataset",
        token: str | None = None,
        cache_dir: str | Path | None = None,
    ):
        """
        Initialize the HF-backed API client.

        :param repo_id: The repo identifier on HF (e.g., "user/dataset"). Eg,
            "BrentLab/yeast_genome_resources"
        :param token: Optional. Not necessary for public repos. May be set via the
            HF_TOKEN environment variable.
        :param repo_type: One of {"model", "dataset", "space"}. Defaults to "dataset".
        :param cache_dir: HF cache_dir for hf_hub_download and snapshot_download (see
            huggingface_hub docs). May be passed via the HF_CACHE_DIR environmental
            variable. If not set, the default HF cache directory is used.
        :raises FileNotFoundError: If the specified cache_dir does not exist.

        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.token = token or os.getenv("HF_TOKEN", None)
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.cache_dir = Path(
            cache_dir if cache_dir else os.getenv("HF_CACHE_DIR", HF_HUB_CACHE)
        )

    @property
    def token(self) -> str | None:
        return self._token

    @token.setter
    def token(self, value: str | None) -> None:
        # TODO: if a token is provided, then validate that it works. Only necessary
        # if token is not None of course
        self._token = value

    @property
    def repo_id(self) -> str:
        return self._repo_id

    @repo_id.setter
    def repo_id(self, value: str) -> None:
        """
        Set the repo_id.

        This setter also calls _get_dataset_size to fetch size info and validate that
        the repo exists and is accessible. No error is raised if the repo is not
        accessible, but an error is logged.

        """
        self._repo_id = value
        try:
            self._get_dataset_size(self._repo_id)
        except (HTTPError, ValueError) as e:
            self.logger.error(f"Could not reach {value}: {e}")

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, value: str | Path) -> None:
        """
        Set the cache directory for huggingface_hub downloads.

        :raises FileNotFoundError: If the specified directory does not exist.

        """
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"Cache directory {path} does not exist")
        self._cache_dir = path

    @property
    def size(self) -> dict[str, Any] | None:
        """
        Size information from the HF Dataset Server API.

        This reaches the /size endpoint. See
        https://github.com/huggingface/dataset-viewer/blob/8f0ae65f0ff64791111d37a725af437c3c752daf/docs/source/size.md

        """
        return getattr(self, "_size", None)

    @size.setter
    def size(self, value: dict[str, Any]) -> None:
        self._size = value

    @property
    def snapshot_path(self) -> Path | None:
        """Path to the last downloaded snapshot (if any)."""
        return getattr(self, "_snapshot_path", None)

    @snapshot_path.setter
    def snapshot_path(self, value: str | Path | None) -> None:
        self._snapshot_path = None if value is None else Path(value)

    def _get_dataset_size_mb(self) -> float:
        """Get dataset size in MB, returning inf if not available."""
        if not self.size:
            return float("inf")
        return (
            self.size.get("size", {})
            .get("dataset", {})
            .get("num_bytes_original_files", float("inf"))
            / MB_TO_BYTES
        )

    def _ensure_str_paths(self, kwargs: dict[str, Any]) -> None:
        """Ensure Path-like arguments are converted to strings."""
        for key in ["local_dir", "cache_dir"]:
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = str(kwargs[key])

    def _build_auth_headers(self) -> dict[str, str]:
        """Build authentication headers if token is available."""
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _normalize_patterns(self, kwargs: dict[str, Any]) -> None:
        """Convert string patterns to lists."""
        for pattern_key in ["allow_patterns", "ignore_patterns"]:
            if pattern_key in kwargs and kwargs[pattern_key] is not None:
                patterns = kwargs[pattern_key]
                if isinstance(patterns, str):
                    kwargs[pattern_key] = [patterns]

    def _get_dataset_size(self, repo_id: str | None = None) -> None:
        """
        Get dataset size information from HuggingFace Dataset Server API.

        :returns: Dict containing size information with additional metadata about
            completeness
        :raises requests.HTTPError: If the API request fails
        :raises ValueError: If the dataset doesn't exist or isn't accessible

        """
        repo_id = repo_id or self.repo_id
        url = f"https://datasets-server.huggingface.co/size?dataset={repo_id}"

        response = requests.get(url, headers=self._build_auth_headers())
        response.raise_for_status()

        data = response.json()

        # Check if size determination was partial
        is_partial = data.get("partial", False)

        if is_partial:
            self.logger.warning(
                f"Size information for {repo_id} is incomplete. "
                "The dataset is too large for complete size determination. "
                "Reported numbers may be lower than actual size."
            )

        # Add metadata about completeness to the response
        if "size" in data and "dataset" in data["size"]:
            data["size"]["dataset"]["size_determination_complete"] = not is_partial
            data["size"]["dataset"]["size_warning"] = (
                "Partial size only - actual dataset may be larger"
                if is_partial
                else "Complete size information"
            )

        self.size = data

    def _download_single_file(
        self,
        filename: str,
        dry_run: bool = False,
        **kwargs,
    ) -> Path:
        """
        Download a single file using hf_hub_download.

        :param filename: File to download
        :param dry_run: If True, log what would be downloaded without downloading
        :param kwargs: Additional arguments passed directly to hf_hub_download
        :return: Path to the downloaded file

        """
        self.logger.info(f"Downloading single file: {filename}")

        if dry_run:
            self.logger.info(f"[DRY RUN] Would download {filename} from {self.repo_id}")
            return Path("dry_run_path")

        # Build base arguments
        hf_kwargs = {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "filename": filename,
            "token": self.token,
            **kwargs,
        }

        # Set cache_dir only if local_dir not specified
        if "local_dir" not in hf_kwargs and self.cache_dir is not None:
            hf_kwargs["cache_dir"] = str(self.cache_dir)

        # Ensure string conversion for Path-like arguments
        self._ensure_str_paths(hf_kwargs)

        file_path = hf_hub_download(**hf_kwargs)
        self._snapshot_path = Path(file_path).parent
        return Path(file_path)

    def _download_snapshot(
        self,
        dry_run: bool = False,
        **kwargs,
    ) -> Path:
        """
        Download repository snapshot using snapshot_download.

        :param dry_run: If True, log what would be downloaded without downloading
        :param kwargs: Additional arguments passed directly to snapshot_download
        :return: Path to the downloaded snapshot

        """
        # Log download plan
        if dry_run:
            self.logger.info(f"[DRY RUN] Would download from {self.repo_id}:")
            self.logger.info(f"  - allow_patterns: {kwargs.get('allow_patterns')}")
            self.logger.info(f"  - ignore_patterns: {kwargs.get('ignore_patterns')}")
            return Path("dry_run_path")

        # Execute snapshot download
        self.logger.info(
            f"Downloading repo snapshot with patterns - "
            f"allow: {kwargs.get('allow_patterns')}, "
            f"ignore: {kwargs.get('ignore_patterns')}"
        )

        # Build base arguments
        # note that kwargs passed into this method will override defaults,
        # including repo_id, etc
        snapshot_kwargs = {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "token": self.token,
            **kwargs,
        }

        # Set cache_dir only if local_dir not specified and cache_dir wasn't passed in
        if (
            "local_dir" not in snapshot_kwargs
            and "cache_dir" not in snapshot_kwargs
            and self.cache_dir is not None
        ):
            snapshot_kwargs["cache_dir"] = str(self.cache_dir)

        # Convert string patterns to lists
        self._normalize_patterns(snapshot_kwargs)

        snapshot_path = snapshot_download(**snapshot_kwargs)
        self.snapshot_path = Path(snapshot_path)
        return self.snapshot_path

    def download(
        self,
        files: list[str] | str | None = None,
        force_full_download: bool = False,
        auto_download_threshold_mb: float = 100.0,
        dry_run: bool = False,
        **kwargs,
    ) -> Path:
        """
        Download dataset by file, patterns or if the dataset is small enough, the entire
        repo.

        :param files: Specific file(s) to download. If provided, uses hf_hub_download
        :param force_full_download: If True, always download entire repo regardless of
            size
        :param auto_download_threshold_mb: Auto-download full repo if estimated size <
            this (MB)
        :param dry_run: If True, log what would be downloaded without actually
            downloading
        :param kwargs: Additional arguments passed to hf_hub_download or
            snapshot_download. Common args: revision, local_dir, cache_dir,
            local_files_only, allow_patterns, ignore_patterns, etc.
        :return: Path to downloaded content (file or directory).

        """
        dataset_size_mb = self._get_dataset_size_mb()
        if dataset_size_mb <= auto_download_threshold_mb or force_full_download:
            self.logger.info(
                f"Dataset size ({dataset_size_mb:.2f} MB) is below the auto-download "
                f"threshold of {auto_download_threshold_mb} MB. Downloading entire "
                "repo."
            )
            files = None
            kwargs.pop("allow_patterns", None)
            kwargs.pop("ignore_patterns", None)
        elif (
            not files
            and not kwargs.get("allow_patterns")
            and not kwargs.get("ignore_patterns")
        ):
            excess_size_mb = dataset_size_mb - auto_download_threshold_mb
            raise RepoTooLargeError(
                f"Dataset size ({dataset_size_mb:.2f} MB) exceeds the "
                f"auto-download threshold of {auto_download_threshold_mb} MB by "
                f"{excess_size_mb:.2f} MB. To download the dataset, either "
                "specify specific files or patterns to download, "
                "set force_full_download=True or increase the "
                "`auto_download_threshold_mb`."
            )
        # Handle specific file downloads
        if files is not None:
            if isinstance(files, str) or (isinstance(files, list) and len(files) == 1):
                # Single file
                filename = files if isinstance(files, str) else files[0]
                self.logger.info(f"Preparing to download single file: {filename}")
                return self._download_single_file(
                    filename=filename, dry_run=dry_run, **kwargs
                )
            elif isinstance(files, list) and len(files) > 1:
                # Multiple files - use snapshot_download with allow_patterns
                if kwargs.get("allow_patterns") is not None:
                    self.logger.warning(
                        "Both 'files' and 'allow_patterns' were provided. "
                        "'files' will take precedence."
                    )
                kwargs["allow_patterns"] = files

        return self._download_snapshot(dry_run=dry_run, **kwargs)

    @abstractmethod
    def parse_datacard(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        """
        Abstract method to parse a datacard from the downloaded content.

        Must be implemented by subclasses.

        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def query(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method to query the API.

        Must be implemented by subclasses.

        """
        raise NotImplementedError("Subclasses must implement this method.")
