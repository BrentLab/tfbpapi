import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import requests
from huggingface_hub import hf_hub_download, repo_info, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from requests import HTTPError


class RepoTooLargeError(ValueError):
    """Raised when repository exceeds auto-download threshold."""

    pass


class AbstractHfAPI(ABC):
    """Abstract base class for creating Hugging Face API clients."""

    # TODO: can revision be set to "latest" by default?
    def __init__(
        self,
        repo_id: str,
        repo_type: Literal["model", "dataset", "space"] = "dataset",
        token: str | None = None,
        cache_dir: str | Path = HF_HUB_CACHE,
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

        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Let user input override env var, but use the env var if available
        resolved_token = token or os.getenv("HF_TOKEN", None)
        resolved_cache_dir = cache_dir or os.getenv("HF_CACHE_DIR", HF_HUB_CACHE)
        if isinstance(resolved_cache_dir, str):
            resolved_cache_dir = Path(resolved_cache_dir)

        self.token = resolved_token
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.cache_dir = resolved_cache_dir

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
        self._repo_id = value
        try:
            self._get_dataset_size(value)
        except (HTTPError, ValueError) as e:
            self.logger.warning(f"Could not validate repo_id {value}: {e}")
            self.logger.info(
                "Repo validation skipped - will be checked on first download"
            )

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, value: str | Path) -> None:
        """Set the cache directory for huggingface_hub downloads."""
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"Cache directory does not exist: {path}")
        self._cache_dir = path

    @property
    def size(self) -> dict[str, Any] | None:
        """Size information from the HF Dataset Server API (if available)."""
        return self._size if hasattr(self, "_size") else None

    @size.setter
    def size(self, value: dict[str, Any]) -> None:
        self._size = value

    @property
    def snapshot_path(self) -> Path | None:
        """Path to the last downloaded snapshot (if any)."""
        return self._snapshot_path if hasattr(self, "_snapshot_path") else None

    @snapshot_path.setter
    def snapshot_path(self, value: str | Path | None) -> None:
        if value is None:
            self._snapshot_path = None
        else:
            self._snapshot_path = Path(value)

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

        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        response = requests.get(url, headers=headers)
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

    def _check_repo_size_and_decide_strategy(
        self,
        auto_download_threshold_mb: float,
        force_full_download: bool,
        allow_patterns: list[str] | str | None,
        ignore_patterns: list[str] | str | None,
        **kwargs,
    ) -> tuple[list[str] | str | None, list[str] | str | None]:
        """
        Check repo size and decide download strategy.

        Returns:
            Tuple of (allow_patterns, ignore_patterns) to use for download

        """
        if force_full_download or auto_download_threshold_mb <= 0:
            return None, None

        try:
            # Get repo info to estimate size
            revision = kwargs.get("revision")
            if revision is not None:
                info = repo_info(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    token=self.token,
                    revision=str(revision),
                )
            else:
                info = repo_info(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    token=self.token,
                )

            # Estimate total size from siblings (files in repo)
            total_size_bytes = sum(
                getattr(sibling, "size", 0) or 0
                for sibling in getattr(info, "siblings", [])
            )
            total_size_mb = total_size_bytes / (1024 * 1024)

            self.logger.info(f"Estimated repo size: {total_size_mb:.2f} MB")

            # If small enough, download everything
            if total_size_mb <= auto_download_threshold_mb:
                self.logger.info(
                    f"Repo size ({total_size_mb:.2f} MB) under threshold "
                    f"({auto_download_threshold_mb} MB), downloading full repo"
                )
                return None, None
            else:
                raise RepoTooLargeError(
                    f"Repo size ({total_size_mb:.2f} MB) exceeds threshold "
                    f"({auto_download_threshold_mb} MB). Use a selective download "
                    "method via `files`, `allow_patterns` or `ignore_patterns`, "
                    "or increase `auto_download_threshold_mb` and try again."
                )

        except Exception as e:
            self.logger.warning(
                f"Could not determine repo size: {e}. Proceeding with "
                f"pattern-based download."
            )
            return allow_patterns, ignore_patterns

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
            **kwargs,  # User kwargs override defaults
        }

        # Set cache_dir only if local_dir not specified
        if "local_dir" not in hf_kwargs and self.cache_dir is not None:
            hf_kwargs["cache_dir"] = str(self.cache_dir)

        # Ensure string conversion for Path-like arguments
        if "local_dir" in hf_kwargs and hf_kwargs["local_dir"] is not None:
            hf_kwargs["local_dir"] = str(hf_kwargs["local_dir"])
        if "cache_dir" in hf_kwargs and hf_kwargs["cache_dir"] is not None:
            hf_kwargs["cache_dir"] = str(hf_kwargs["cache_dir"])

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
        # by user
        if (
            "local_dir" not in snapshot_kwargs
            and "cache_dir" not in snapshot_kwargs
            and self.cache_dir is not None
        ):
            snapshot_kwargs["cache_dir"] = str(self.cache_dir)

        # if allow_patterns or ignore_patterns are strings, convert to list
        for pattern_key in ["allow_patterns", "ignore_patterns"]:
            if (
                pattern_key in snapshot_kwargs
                and snapshot_kwargs[pattern_key] is not None
            ):
                patterns = snapshot_kwargs[pattern_key]
                if isinstance(patterns, str):
                    snapshot_kwargs[pattern_key] = [patterns]

        snapshot_path = snapshot_download(**snapshot_kwargs)
        self.snapshot_path = Path(snapshot_path)
        return self.snapshot_path

    def download(
        self,
        *,
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
        # Handle single file download
        if files is not None:
            if isinstance(files, str):
                files = [files]

            if len(files) == 1:
                return self._download_single_file(
                    filename=files[0],
                    dry_run=dry_run,
                    **kwargs,
                )
            else:
                # Multiple specific files - use filtered snapshot_download
                self.logger.info(f"Downloading specific files: {files}")
                if kwargs.get("allow_patterns") is not None:
                    self.logger.warning(
                        "`allow_patterns` will be overridden by `files` argument"
                    )
                kwargs["allow_patterns"] = files

        # Check repo size and adjust download strategy if needed
        allow_patterns, ignore_patterns = self._check_repo_size_and_decide_strategy(
            auto_download_threshold_mb=auto_download_threshold_mb,
            force_full_download=force_full_download,
            allow_patterns=kwargs.get("allow_patterns"),
            ignore_patterns=kwargs.get("ignore_patterns"),
            **kwargs,
        )

        # Update kwargs with determined patterns
        if allow_patterns is not None:
            kwargs["allow_patterns"] = allow_patterns
        if ignore_patterns is not None:
            kwargs["ignore_patterns"] = ignore_patterns

        # Execute snapshot download
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
