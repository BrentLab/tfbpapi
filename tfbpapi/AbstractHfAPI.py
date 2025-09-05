import logging
import os
from pathlib import Path
from typing import Any, Iterable, Mapping
import hashlib
import json

from .ParamsDict import ParamsDict


class AbstractHfAPI:
    """ 
    Abstract base class for creating Hugging Face API clients.
    """


    def __init__(
        self,
        repo_id: str = "",
        repo_type: str | None = "dataset",
        revision: str | None = None,
        token: str = "",
        cache_dir: str | Path | None = None,
        local_dir: str | Path | None = None,
        **kwargs,
    ):
        """
        Initialize the HF-backed API client.

        :param repo_id: The repo identifier on HF (e.g., "user/dataset").
        :param repo_type: One of {"model", "dataset", "space"}. Defaults to "dataset".
        :param revision: Optional git revision (branch, tag, or commit SHA).
        :param token: Authentication token. Defaults to env `HF_TOKEN` or `TOKEN`.
        :param cache_dir: HF cache dir; passed to snapshot_download.
        :param local_dir: Optional local materialization dir; if supported by the
            installed `huggingface_hub`, downloaded files will be placed here.
        :param kwargs: Additional keyword arguments for ParamsDict construction.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self._token = token or os.getenv("HF_TOKEN")
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.revision = revision
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.local_dir = Path(local_dir) if local_dir is not None else None
        self._snapshot_path: Path | None = None

        self.params = ParamsDict(
            params=kwargs.pop("params", {}),
            valid_keys=kwargs.pop("valid_keys", []),
        )

        self._result_cache: dict[str, dict[str, Any]] = {}

    @property
    def token(self) -> str:
        return self._token

    @token.setter
    def token(self, value: str) -> None:
        self._token = value

    def _hash_params(self, params: Mapping[str, Any] | None) -> str:
        """Stable hash for query-result caching across identical filters.

        Incorporate repo identifiers to avoid cross-repo collisions.
        """
        payload = {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "revision": self.revision,
            "params": params or {},
        }
        data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(data).hexdigest()

    def ensure_snapshot(
        self,
        *,
        allow_patterns: Iterable[str] | None = None,
        ignore_patterns: Iterable[str] | None = None,
        local_files_only: bool = False,
    ) -> Path:
        """
        Ensure a local snapshot exists using HF cache system and return its path.
        """
        try:
            # Lazy import to avoid hard dependency during package import
            from huggingface_hub import snapshot_download  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "huggingface_hub is required to use AbstractHfAPI.ensure_snapshot()"
            ) from e

        kwargs: dict[str, Any] = {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "revision": self.revision,
            "cache_dir": str(self.cache_dir) if self.cache_dir is not None else None,
            "allow_patterns": list(allow_patterns) if allow_patterns else None,
            "ignore_patterns": list(ignore_patterns) if ignore_patterns else None,
            "local_files_only": local_files_only,
            "token": self.token or None,
        }

        if self.local_dir is not None:
            kwargs_with_local = dict(kwargs)
            kwargs_with_local["local_dir"] = str(self.local_dir)
            try:
                snapshot = snapshot_download(**kwargs_with_local)  # type: ignore[arg-type]
            except TypeError:
                self.logger.info(
                    "Installed huggingface_hub does not support local_dir; retrying without it"
                )
                snapshot = snapshot_download(**kwargs)  # type: ignore[arg-type]
        else:
            snapshot = snapshot_download(**kwargs)  # type: ignore[arg-type]

        self._snapshot_path = Path(snapshot)
        return self._snapshot_path

    def fetch_repo_metadata(self, *, filename_candidates: Iterable[str] | None = None) -> dict[str, Any] | None:
        """
        Fetch lightweight metadata for a file in the repo using get_hf_file_metadata.

        Tries a list of candidate files (e.g., README.md, dataset_infos.json). Returns
        a dict with keys {commit_hash, etag, location, filename} or None if none
        of the candidates exist.
        """
        try:
            from huggingface_hub import get_hf_file_metadata, hf_hub_url  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "huggingface_hub is required to use AbstractHfAPI.fetch_repo_metadata()"
            ) from e

        candidates: list[str] = list(filename_candidates or [
            "README.md",
            "README.MD",
            "README.rst",
            "README.txt",
            "dataset_infos.json",
        ])

        for fname in candidates:
            try:
                url = hf_hub_url(
                    repo_id=self.repo_id,
                    filename=fname,
                    repo_type=self.repo_type,
                    revision=self.revision,
                )
                meta = get_hf_file_metadata(url=url, token=self.token or None)
                return {
                    "commit_hash": getattr(meta, "commit_hash", None),
                    "etag": getattr(meta, "etag", None),
                    "location": getattr(meta, "location", None),
                    "filename": fname,
                }
            except Exception as e:  # EntryNotFoundError, RepositoryNotFoundError, etc.
                self.logger.debug(f"Metadata not found for {fname}: {e}")
                continue
        return None

    def open_dataset(
        self,
        snapshot_path: str | Path | None = None,
        *,
        files: Iterable[str | Path] | None = None,
        format: str | None = None,
        partitioning: str | None = "hive",
    ):
        """
        Build a pyarrow.dataset from the local snapshot.

        :param snapshot_path: Path to the HF snapshot (defaults to last ensured).
        :param files: Optional iterable of files to include explicitly.
        :param format: Optional format hint (e.g., "parquet").
        :param partitioning: Optional partitioning strategy (default: "hive").
        :return: A pyarrow.dataset.Dataset instance.
        """
        try:
            import pyarrow.dataset as ds  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "pyarrow is required to use AbstractHfAPI.open_dataset()"
            ) from e

        base_path = Path(snapshot_path) if snapshot_path is not None else self._snapshot_path
        if base_path is None:
            raise RuntimeError("Snapshot not ensured. Call ensure_snapshot() first.")

        if files is not None:
            file_list = [str(Path(f)) for f in files]
            dataset = ds.dataset(file_list, format=format or None, partitioning=partitioning)
        else:
            dataset = ds.dataset(str(base_path), format=format or None, partitioning=partitioning)
        return dataset

    def build_query(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """
        Translate user params to dataset-specific query plan.

        Expected keys in the returned dict (subclasses should implement):
        - file_patterns: Optional[list[str]] of glob patterns relative to snapshot root
        - filter: Optional[pyarrow.dataset.Expression] to pushdown into scans
        - format: Optional[str] dataset format hint (e.g., "parquet")
        """
        raise NotImplementedError(
            f"`build_query()` is not implemented for {self.__class__.__name__}"
        )

    def read(self, params: Mapping[str, Any] | None = None, **kwargs) -> dict[str, Any]:
        """
        Execute a read using HF snapshot + optional dataset scanning.

        Returns a dict with keys: {"metadata": <dict|None>, "data": <pyarrow.Table>}.
        """
        params = params or {}
        cache_key = self._hash_params(params)
        if cache_key in self._result_cache:
            self.logger.debug("Returning cached query result")
            return self._result_cache[cache_key]

        # Build query plan from params
        plan = self.build_query(params)
        file_patterns: Iterable[str] | None = plan.get("file_patterns")
        filter_expr = plan.get("filter")
        fmt_hint: str | None = plan.get("format") or kwargs.get("format")

        # Optionally restrict download via allow_patterns; otherwise download all
        snapshot_path = self.ensure_snapshot(
            allow_patterns=file_patterns,
        )

        # If patterns were used, build the explicit file list under the snapshot
        files: list[str] | None = None
        if file_patterns:
            files = []
            for patt in file_patterns:
                files.extend([str(p) for p in Path(snapshot_path).rglob(patt)])

        dataset = self.open_dataset(snapshot_path=snapshot_path, files=files, format=fmt_hint)

        # Materialize to a pyarrow.Table; subclasses can choose filter types
        try:
            table = dataset.to_table(filter=filter_expr) if filter_expr is not None else dataset.to_table()
        except Exception as e:
            # Add context then re-raise
            self.logger.error(f"Failed to scan dataset with provided filter: {e}")
            raise

        metadata = self.fetch_repo_metadata()
        result = {"metadata": metadata, "data": table}

        # Memoize
        self._result_cache[cache_key] = result
        return result

    def __call__(self, *args, **kwargs):
        # Delegate to read() for ergonomic usage
        params = args[0] if args else kwargs.pop("params", None)
        return self.read(params=params or {}, **kwargs)