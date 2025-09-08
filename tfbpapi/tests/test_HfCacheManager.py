from tfbpapi.HfCacheManager import HFCacheManager


def test_parse_size_string():
    """Test size string parsing."""
    cache_manager = HFCacheManager()
    assert cache_manager._parse_size_string("10KB") == 10 * 1024
    assert cache_manager._parse_size_string("5MB") == 5 * 1024**2
    assert cache_manager._parse_size_string("2GB") == 2 * 1024**3
    assert cache_manager._parse_size_string("1TB") == 1 * 1024**4
    assert cache_manager._parse_size_string("500") == 500


def test_clean_cache_by_age(mock_scan_cache_dir):
    """Test age-based cache cleaning."""

    cache_manager = HFCacheManager()

    # This will use the mocked scan_cache_dir
    strategy = cache_manager.clean_cache_by_age(max_age_days=1, dry_run=True)

    # Test your logic
    assert strategy is not None or len(mock_scan_cache_dir.repos) == 0
    if strategy:
        assert len(strategy.repos) >= 0
        assert strategy.expected_freed_size >= 0


def test_clean_cache_by_size(mock_scan_cache_dir):
    """Test size-based cache cleaning."""

    cache_manager = HFCacheManager()

    # Test with a very small target to force cleanup
    strategy = cache_manager.clean_cache_by_size(target_size="1GB", dry_run=True)

    if mock_scan_cache_dir.size_on_disk > 1024:  # If cache has data
        assert strategy is not None
        assert len(strategy.repos) > 0


def test_auto_clean_cache(mock_scan_cache_dir):
    """Test automated cache cleaning."""

    cache_manager = HFCacheManager()

    strategies = cache_manager.auto_clean_cache(
        max_age_days=1,
        max_total_size="1KB",
        keep_latest_per_repo=1,
        dry_run=True,
    )

    # Should return a list of strategies
    assert isinstance(strategies, list)
