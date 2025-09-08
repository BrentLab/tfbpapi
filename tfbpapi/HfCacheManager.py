import logging
from datetime import datetime, timedelta
from typing import Literal

from huggingface_hub import scan_cache_dir
from huggingface_hub.utils import DeleteCacheStrategy


class HFCacheManager:
    """Cache memory management for Hugging Face Hub cache."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

    def clean_cache_by_age(
        self,
        max_age_days: int = 30,
        dry_run: bool = True,
    ) -> DeleteCacheStrategy:
        """
        Clean cache entries older than specified age.

        :param max_age_days: Remove revisions older than this many days
        :param  dry_run: If True, show what would be deleted without executing
            size_threshold: Only delete if total cache size exceeds this (e.g., "10GB")

        :return: DeleteCacheStrategy object that can be executed

        """
        cache_info = scan_cache_dir()
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        old_revisions = []
        for repo in cache_info.repos:
            for revision in repo.revisions:
                # Check if revision is older than cutoff
                revision_date = datetime.fromtimestamp(revision.last_modified)
                if revision_date < cutoff_date:
                    old_revisions.append(revision.commit_hash)
                    self.logger.debug(
                        f"Marking for deletion: {revision.commit_hash} "
                        f"(last modified: {revision.last_modified})"
                    )

        if not old_revisions:
            self.logger.info("No old revisions found to delete")
            # return None

        delete_strategy = cache_info.delete_revisions(*old_revisions)

        self.logger.info(
            f"Found {len(old_revisions)} old revisions. "
            f"Will free {delete_strategy.expected_freed_size_str}"
        )

        if not dry_run:
            delete_strategy.execute()
            self.logger.info(
                f"Cache cleanup completed. Freed "
                f"{delete_strategy.expected_freed_size_str}"
            )
        else:
            self.logger.info("Dry run completed. Use dry_run=False to execute deletion")

        return delete_strategy

    def clean_cache_by_size(
        self,
        target_size: str,
        strategy: Literal[
            "oldest_first", "largest_first", "least_used"
        ] = "oldest_first",
        dry_run: bool = True,
    ) -> DeleteCacheStrategy:
        """
        Clean cache to reach target size by removing revisions.

        :param target_size: Target cache size (e.g., "5GB", "500MB")
        :param strategy: Deletion strategy - "oldest_first", "largest_first",
            "least_used"
        :param dry_run: If True, show what would be deleted without executing

        :return: DeleteCacheStrategy object that can be executed

        """
        cache_info = scan_cache_dir()
        current_size = cache_info.size_on_disk
        target_bytes = self._parse_size_string(target_size)

        if current_size <= target_bytes:
            self.logger.info(
                f"Cache size ({cache_info.size_on_disk_str}) already below "
                f"target ({target_size})"
            )

        bytes_to_free = current_size - target_bytes

        # Get all revisions sorted by strategy
        all_revisions = []
        for repo in cache_info.repos:
            for revision in repo.revisions:
                all_revisions.append(revision)

        # Sort revisions based on strategy
        if strategy == "oldest_first":
            all_revisions.sort(key=lambda r: r.last_modified)
        elif strategy == "largest_first":
            all_revisions.sort(key=lambda r: r.size_on_disk, reverse=True)
        elif strategy == "least_used":
            # Use last_modified as proxy for usage
            all_revisions.sort(key=lambda r: r.last_modified)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Select revisions to delete
        revisions_to_delete = []
        freed_bytes = 0

        for revision in all_revisions:
            if freed_bytes >= bytes_to_free:
                break
            revisions_to_delete.append(revision.commit_hash)
            freed_bytes += revision.size_on_disk

        if not revisions_to_delete:
            self.logger.warning("No revisions selected for deletion")

        delete_strategy = cache_info.delete_revisions(*revisions_to_delete)

        self.logger.info(
            f"Selected {len(revisions_to_delete)} revisions for deletion. "
            f"Will free {delete_strategy.expected_freed_size_str}"
        )

        if not dry_run:
            delete_strategy.execute()
            self.logger.info(
                f"Cache cleanup completed. Freed "
                f"{delete_strategy.expected_freed_size_str}"
            )
        else:
            self.logger.info("Dry run completed. Use dry_run=False to execute deletion")

        return delete_strategy

    def clean_unused_revisions(
        self, keep_latest: int = 2, dry_run: bool = True
    ) -> DeleteCacheStrategy:
        """
        Clean unused revisions, keeping only the latest N revisions per repo.

        :param keep_latest: Number of latest revisions to keep per repo
        :param dry_run: If True, show what would be deleted without executing
        :return: DeleteCacheStrategy object that can be executed

        """
        cache_info = scan_cache_dir()
        revisions_to_delete = []

        for repo in cache_info.repos:
            # Sort revisions by last modified (newest first)
            sorted_revisions = sorted(
                repo.revisions, key=lambda r: r.last_modified, reverse=True
            )

            # Keep the latest N, mark the rest for deletion
            if len(sorted_revisions) > keep_latest:
                old_revisions = sorted_revisions[keep_latest:]
                for revision in old_revisions:
                    revisions_to_delete.append(revision.commit_hash)
                    self.logger.debug(
                        f"Marking old revision for deletion: {repo.repo_id} - "
                        f"{revision.commit_hash}"
                    )

        delete_strategy = cache_info.delete_revisions(*revisions_to_delete)

        self.logger.info(
            f"Found {len(revisions_to_delete)} unused revisions. "
            f"Will free {delete_strategy.expected_freed_size_str}"
        )

        if not dry_run:
            delete_strategy.execute()
            self.logger.info(
                f"Cache cleanup completed. Freed "
                f"{delete_strategy.expected_freed_size_str}"
            )
        else:
            self.logger.info("Dry run completed. Use dry_run=False to execute deletion")

        return delete_strategy

    def auto_clean_cache(
        self,
        max_age_days: int = 30,
        max_total_size: str = "10GB",
        keep_latest_per_repo: int = 2,
        dry_run: bool = True,
    ) -> list[DeleteCacheStrategy]:
        """
        Automated cache cleaning with multiple strategies.

        :param max_age_days: Remove revisions older than this
        :param max_total_size: Target maximum cache size
        :param keep_latest_per_repo: Keep this many latest revisions per repo
        :param dry_run: If True, show what would be deleted without executing
        :return: List of DeleteCacheStrategy objects that were executed

        """
        strategies_executed = []

        self.logger.info("Starting automated cache cleanup...")

        # Step 1: Remove very old revisions
        strategy = self.clean_cache_by_age(max_age_days=max_age_days, dry_run=dry_run)
        if strategy:
            strategies_executed.append(strategy)

        # Step 2: Remove unused revisions (keep only latest per repo)
        strategy = self.clean_unused_revisions(
            keep_latest=keep_latest_per_repo, dry_run=dry_run
        )
        if strategy:
            strategies_executed.append(strategy)

        # Step 3: If still over size limit, remove more aggressively
        cache_info = scan_cache_dir()
        if cache_info.size_on_disk > self._parse_size_string(max_total_size):
            strategy = self.clean_cache_by_size(
                target_size=max_total_size, strategy="oldest_first", dry_run=dry_run
            )
            if strategy:
                strategies_executed.append(strategy)

        total_freed = sum(s.expected_freed_size for s in strategies_executed)
        self.logger.info(
            f"Automated cleanup complete. Total freed: "
            f"{self._format_bytes(total_freed)}"
        )

        return strategies_executed

    def _parse_size_string(self, size_str: str) -> int:
        """Parse size string like '10GB' to bytes."""
        size_str = size_str.upper().strip()

        # Check longer units first to avoid partial matches
        multipliers = {"TB": 1024**4, "GB": 1024**3, "MB": 1024**2, "KB": 1024, "B": 1}

        for unit, multiplier in multipliers.items():
            if size_str.endswith(unit):
                number = float(size_str[: -len(unit)])
                return int(number * multiplier)

        # If no unit specified, assume bytes
        return int(size_str)

    def _format_bytes(self, bytes_size: int) -> str:
        """Format bytes into human readable string."""
        if bytes_size == 0:
            return "0B"

        # iterate over common units, dividing by 1024 each time, to find an
        # appropriate unit. Default to TB if the size is very large
        size = float(bytes_size)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}TB"
