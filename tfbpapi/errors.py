"""Custom exception classes for dataset management."""

from typing import Any


class DatasetError(Exception):
    """Base exception for all dataset-related errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}

    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{base_msg} (Details: {detail_str})"
        return base_msg


class RepoTooLargeError(DatasetError):
    """Raised when repository exceeds auto-download threshold."""

    def __init__(self, repo_id: str, size_mb: float, threshold_mb: float):
        message = f"Repository '{repo_id}' is too large for "
        f"auto-download: {size_mb:.2f}MB exceeds {threshold_mb}MB threshold"
        super().__init__(
            message,
            details={
                "repo_id": repo_id,
                "actual_size_mb": size_mb,
                "threshold_mb": threshold_mb,
            },
        )
        self.repo_id = repo_id
        self.size_mb = size_mb
        self.threshold_mb = threshold_mb


class DataCardParsingError(DatasetError):
    """Raised when dataset card parsing fails."""

    def __init__(
        self,
        message: str,
        repo_id: str | None = None,
        config_name: str | None = None,
        original_error: Exception | None = None,
    ):
        details: dict[str, Any] = {}
        if repo_id:
            details["repo_id"] = repo_id
        if config_name:
            details["config_name"] = config_name
        if original_error:
            details["original_error"] = str(original_error)

        super().__init__(message, details)
        self.repo_id = repo_id
        self.config_name = config_name
        self.original_error = original_error


class HfDataFetchError(DatasetError):
    """Raised when HuggingFace API requests fail."""

    def __init__(
        self,
        message: str,
        repo_id: str | None = None,
        status_code: int | None = None,
        endpoint: str | None = None,
    ):
        details: dict[str, Any] = {}
        if repo_id:
            details["repo_id"] = repo_id
        if status_code:
            details["status_code"] = status_code
        if endpoint:
            details["endpoint"] = endpoint

        super().__init__(message, details)
        self.repo_id = repo_id
        self.status_code = status_code
        self.endpoint = endpoint


class TableNotFoundError(DatasetError):
    """Raised when requested table doesn't exist."""

    def __init__(self, table_name: str, available_tables: list | None = None):
        available_str = (
            f"Available tables: {available_tables}"
            if available_tables
            else "No tables available"
        )
        message = f"Table '{table_name}' not found. {available_str}"

        super().__init__(
            message,
            details={
                "requested_table": table_name,
                "available_tables": available_tables or [],
            },
        )
        self.table_name = table_name
        self.available_tables = available_tables or []


class MissingDatasetTypeError(DatasetError):
    """Raised when dataset_type field is missing from config."""

    def __init__(self, config_name: str, available_fields: list | None = None):
        fields_str = f"Available fields: {available_fields}" if available_fields else ""
        message = (
            f"Missing 'dataset_type' field in config '{config_name}'. {fields_str}"
        )

        super().__init__(
            message,
            details={
                "config_name": config_name,
                "available_fields": available_fields or [],
            },
        )
        self.config_name = config_name
        self.available_fields = available_fields or []


class InvalidDatasetTypeError(DatasetError):
    """Raised when dataset_type value is not recognized."""

    def __init__(self, invalid_type: str, valid_types: list | None = None):
        valid_str = f"Valid types: {valid_types}" if valid_types else ""
        message = f"Invalid dataset type '{invalid_type}'. {valid_str}"

        super().__init__(
            message,
            details={"invalid_type": invalid_type, "valid_types": valid_types or []},
        )
        self.invalid_type = invalid_type
        self.valid_types = valid_types or []


class ConfigNotFoundError(DatasetError):
    """Raised when a requested config doesn't exist."""

    def __init__(
        self,
        config_name: str,
        repo_id: str | None = None,
        available_configs: list | None = None,
    ):
        repo_str = f" in repository '{repo_id}'" if repo_id else ""
        available_str = (
            f"Available configs: {available_configs}" if available_configs else ""
        )
        message = f"Config '{config_name}' not found{repo_str}. {available_str}"

        super().__init__(
            message,
            details={
                "config_name": config_name,
                "repo_id": repo_id,
                "available_configs": available_configs or [],
            },
        )
        self.config_name = config_name
        self.repo_id = repo_id
        self.available_configs = available_configs or []


class DataCardError(DatasetError):
    """Base exception for DataCard operations."""

    pass


class DataCardValidationError(DataCardError):
    """Exception raised when dataset card validation fails."""

    def __init__(
        self,
        message: str,
        repo_id: str | None = None,
        validation_errors: list | None = None,
    ):
        details: dict[str, Any] = {}
        if repo_id:
            details["repo_id"] = repo_id
        if validation_errors:
            details["validation_errors"] = validation_errors

        super().__init__(message, details)
        self.repo_id = repo_id
        self.validation_errors = validation_errors or []


class DataCardMetadataError(DataCardError):
    """Exception raised when metadata extraction fails."""

    def __init__(
        self,
        message: str,
        config_name: str | None = None,
        field_name: str | None = None,
    ):
        details: dict[str, Any] = {}
        if config_name:
            details["config_name"] = config_name
        if field_name:
            details["field_name"] = field_name

        super().__init__(message, details)
        self.config_name = config_name
        self.field_name = field_name


class InvalidFilterFieldError(DatasetError):
    """Raised when filter fields don't exist in metadata columns."""

    def __init__(
        self,
        config_name: str,
        invalid_fields: list[str],
        available_fields: list[str] | None = None,
    ):
        invalid_str = ", ".join(f"'{field}'" for field in invalid_fields)
        available_str = (
            f"Available fields: {sorted(available_fields)}"
            if available_fields
            else "No fields available"
        )
        message = (
            f"Invalid filter field(s) {invalid_str} for config '{config_name}'. "
            f"{available_str}"
        )

        super().__init__(
            message,
            details={
                "config_name": config_name,
                "invalid_fields": invalid_fields,
                "available_fields": (
                    sorted(available_fields) if available_fields else []
                ),
            },
        )
        self.config_name = config_name
        self.invalid_fields = invalid_fields
        self.available_fields = sorted(available_fields) if available_fields else []
