"""
Tests for VirtualDB unified query interface.

Tests configuration loading, schema discovery, querying, filtering, and caching.

"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml  # type: ignore

from tfbpapi.virtual_db import VirtualDB, get_nested_value, normalize_value


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_nested_value_simple(self):
        """Test simple nested dict navigation."""
        data = {"media": {"name": "YPD"}}
        result = get_nested_value(data, "media.name")
        assert result == "YPD"

    def test_get_nested_value_missing_key(self):
        """Test that missing keys return None."""
        data = {"media": {"name": "YPD"}}
        result = get_nested_value(data, "media.carbon_source")
        assert result is None

    def test_get_nested_value_list_extraction(self):
        """Test extracting property from list of dicts."""
        data = {
            "media": {
                "carbon_source": [{"compound": "glucose"}, {"compound": "galactose"}]
            }
        }
        result = get_nested_value(data, "media.carbon_source.compound")
        assert result == ["glucose", "galactose"]

    def test_get_nested_value_non_dict(self):
        """Test that non-dict input returns None."""
        result = get_nested_value("not a dict", "path")  # type: ignore
        assert result is None

    def test_normalize_value_exact_match(self):
        """Test exact alias match."""
        aliases = {"glucose": ["D-glucose", "dextrose"]}
        result = normalize_value("D-glucose", aliases)
        assert result == "glucose"

    def test_normalize_value_case_insensitive(self):
        """Test case-insensitive matching."""
        aliases = {"glucose": ["D-glucose", "dextrose"]}
        result = normalize_value("DEXTROSE", aliases)
        assert result == "glucose"

    def test_normalize_value_no_match(self):
        """Test pass-through when no alias matches."""
        aliases = {"glucose": ["D-glucose"]}
        result = normalize_value("maltose", aliases)
        assert result == "maltose"

    def test_normalize_value_no_aliases(self):
        """Test pass-through when no aliases provided."""
        result = normalize_value("D-glucose", None)
        assert result == "D-glucose"

    def test_normalize_value_missing_value_label(self):
        """Test missing value handling."""
        result = normalize_value(None, None, "unspecified")
        assert result == "unspecified"

    def test_normalize_value_missing_value_no_label(self):
        """Test missing value without label."""
        result = normalize_value(None, None)
        assert result == "None"


class TestVirtualDBConfig:
    """Tests for VirtualDB configuration loading."""

    def create_test_config(self, **overrides):
        """Helper to create test configuration file."""
        config = {
            "factor_aliases": {
                "carbon_source": {
                    "glucose": ["D-glucose", "dextrose"],
                    "galactose": ["D-galactose", "Galactose"],
                }
            },
            "missing_value_labels": {"carbon_source": "unspecified"},
            "description": {"carbon_source": "Carbon source in growth media"},
            "repositories": {
                "BrentLab/test_repo": {
                    "temperature_celsius": {"path": "temperature_celsius"},
                    "dataset": {
                        "test_dataset": {
                            "carbon_source": {
                                "field": "condition",
                                "path": "media.carbon_source.compound",
                            }
                        }
                    },
                }
            },
        }
        config.update(overrides)
        return config

    def test_init_with_valid_config(self):
        """Test VirtualDB initialization with valid config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(self.create_test_config(), f)
            config_path = f.name

        try:
            vdb = VirtualDB(config_path)
            assert vdb.config is not None
            assert vdb.token is None
            assert len(vdb.cache) == 0
        finally:
            Path(config_path).unlink()

    def test_init_with_token(self):
        """Test VirtualDB initialization with HF token."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(self.create_test_config(), f)
            config_path = f.name

        try:
            vdb = VirtualDB(config_path, token="test_token")
            assert vdb.token == "test_token"
        finally:
            Path(config_path).unlink()

    def test_init_missing_config_file(self):
        """Test error when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            VirtualDB("/nonexistent/path.yaml")

    def test_repr(self):
        """Test string representation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(self.create_test_config(), f)
            config_path = f.name

        try:
            vdb = VirtualDB(config_path)
            repr_str = repr(vdb)
            assert "VirtualDB" in repr_str
            assert "1 repositories" in repr_str
            assert "1 datasets configured" in repr_str
            assert "0 views cached" in repr_str
        finally:
            Path(config_path).unlink()


class TestSchemaDiscovery:
    """Tests for schema discovery methods."""

    def create_multi_dataset_config(self):
        """Create config with multiple datasets."""
        return {
            "factor_aliases": {},
            "repositories": {
                "BrentLab/repo1": {
                    "temperature_celsius": {"path": "temperature_celsius"},
                    "dataset": {
                        "dataset1": {
                            "carbon_source": {
                                "field": "condition",
                                "path": "media.carbon_source",
                            }
                        }
                    },
                },
                "BrentLab/repo2": {
                    "nitrogen_source": {"path": "media.nitrogen_source"},
                    "dataset": {
                        "dataset2": {
                            "carbon_source": {"path": "media.carbon_source"},
                            "temperature_celsius": {"path": "temperature_celsius"},
                        }
                    },
                },
            },
        }

    def test_get_fields_all_datasets(self):
        """Test getting all fields across all datasets."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(self.create_multi_dataset_config(), f)
            config_path = f.name

        try:
            vdb = VirtualDB(config_path)
            fields = vdb.get_fields()
            assert "carbon_source" in fields
            assert "temperature_celsius" in fields
            assert "nitrogen_source" in fields
            assert fields == sorted(fields)  # Should be sorted
        finally:
            Path(config_path).unlink()

    def test_get_fields_specific_dataset(self):
        """Test getting fields for specific dataset."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(self.create_multi_dataset_config(), f)
            config_path = f.name

        try:
            vdb = VirtualDB(config_path)
            fields = vdb.get_fields("BrentLab/repo1", "dataset1")
            assert "carbon_source" in fields
            assert "temperature_celsius" in fields
            # nitrogen_source is in repo2, not repo1
            assert "nitrogen_source" not in fields
        finally:
            Path(config_path).unlink()

    def test_get_fields_invalid_partial_args(self):
        """Test error when only one of repo_id/config_name provided."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(self.create_multi_dataset_config(), f)
            config_path = f.name

        try:
            vdb = VirtualDB(config_path)
            with pytest.raises(ValueError, match="Both repo_id and config_name"):
                vdb.get_fields(repo_id="BrentLab/repo1")
        finally:
            Path(config_path).unlink()

    def test_get_common_fields(self):
        """Test getting fields common to all datasets."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(self.create_multi_dataset_config(), f)
            config_path = f.name

        try:
            vdb = VirtualDB(config_path)
            common = vdb.get_common_fields()
            # Both datasets have carbon_source and temperature_celsius
            assert "carbon_source" in common
            assert "temperature_celsius" in common
            # nitrogen_source is only in repo2
            assert "nitrogen_source" not in common
        finally:
            Path(config_path).unlink()


class TestCaching:
    """Tests for view materialization and caching."""

    def create_simple_config(self):
        """Create simple config for testing."""
        return {
            "factor_aliases": {},
            "repositories": {
                "BrentLab/test_repo": {
                    "dataset": {
                        "test_dataset": {
                            "carbon_source": {"path": "media.carbon_source"}
                        }
                    }
                }
            },
        }

    def test_invalidate_cache_all(self):
        """Test invalidating all cache."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(self.create_simple_config(), f)
            config_path = f.name

        try:
            vdb = VirtualDB(config_path)
            # Manually add to cache
            vdb.cache[("BrentLab/test_repo", "test_dataset")] = pd.DataFrame()
            assert len(vdb.cache) == 1

            vdb.invalidate_cache()
            assert len(vdb.cache) == 0
        finally:
            Path(config_path).unlink()

    def test_invalidate_cache_specific(self):
        """Test invalidating specific dataset cache."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(self.create_simple_config(), f)
            config_path = f.name

        try:
            vdb = VirtualDB(config_path)
            # Add multiple entries to cache
            vdb.cache[("BrentLab/test_repo", "test_dataset")] = pd.DataFrame()
            vdb.cache[("BrentLab/other_repo", "other_dataset")] = pd.DataFrame()
            assert len(vdb.cache) == 2

            vdb.invalidate_cache([("BrentLab/test_repo", "test_dataset")])
            assert len(vdb.cache) == 1
            assert ("BrentLab/other_repo", "other_dataset") in vdb.cache
        finally:
            Path(config_path).unlink()


class TestFiltering:
    """Tests for filter application logic."""

    def test_apply_filters_exact_match(self):
        """Test exact value matching in filters."""
        df = pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3"],
                "carbon_source": ["glucose", "galactose", "glucose"],
            }
        )

        # Create minimal VirtualDB instance
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "repositories": {
                    "BrentLab/test": {
                        "dataset": {
                            "test": {"carbon_source": {"path": "media.carbon_source"}}
                        }
                    }
                }
            }
            yaml.dump(config, f)
            config_path = f.name

        try:
            vdb = VirtualDB(config_path)
            filtered = vdb._apply_filters(
                df, {"carbon_source": "glucose"}, "BrentLab/test", "test"
            )
            assert len(filtered) == 2
            assert all(filtered["carbon_source"] == "glucose")
        finally:
            Path(config_path).unlink()

    def test_apply_filters_numeric_range(self):
        """Test numeric range filtering."""
        df = pd.DataFrame(
            {"sample_id": ["s1", "s2", "s3"], "temperature_celsius": [25, 30, 37]}
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "repositories": {
                    "BrentLab/test": {
                        "dataset": {
                            "test": {
                                "temperature_celsius": {"path": "temperature_celsius"}
                            }
                        }
                    }
                }
            }
            yaml.dump(config, f)
            config_path = f.name

        try:
            vdb = VirtualDB(config_path)

            # Test >= operator
            filtered = vdb._apply_filters(
                df, {"temperature_celsius": (">=", 30)}, "BrentLab/test", "test"
            )
            assert len(filtered) == 2
            assert all(filtered["temperature_celsius"] >= 30)

            # Test between operator
            filtered = vdb._apply_filters(
                df,
                {"temperature_celsius": ("between", 28, 32)},
                "BrentLab/test",
                "test",
            )
            assert len(filtered) == 1
            assert filtered.iloc[0]["temperature_celsius"] == 30
        finally:
            Path(config_path).unlink()

    def test_apply_filters_with_alias_expansion(self):
        """Test filter with alias expansion."""
        df = pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3"],
                "carbon_source": ["glucose", "D-glucose", "galactose"],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "factor_aliases": {
                    "carbon_source": {"glucose": ["D-glucose", "dextrose", "glucose"]}
                },
                "repositories": {
                    "BrentLab/test": {
                        "dataset": {
                            "test": {"carbon_source": {"path": "media.carbon_source"}}
                        }
                    }
                },
            }
            yaml.dump(config, f)
            config_path = f.name

        try:
            vdb = VirtualDB(config_path)
            filtered = vdb._apply_filters(
                df, {"carbon_source": "glucose"}, "BrentLab/test", "test"
            )
            # Should match both "glucose" and "D-glucose" due to alias expansion
            assert len(filtered) == 2
        finally:
            Path(config_path).unlink()


class TestExtraction:
    """Tests for metadata extraction methods."""

    def test_add_field_metadata(self):
        """Test adding field-level metadata to DataFrame."""
        df = pd.DataFrame({"sample_id": ["s1", "s2"], "condition": ["YPD", "YPG"]})

        field_metadata = {
            "YPD": {"carbon_source": ["glucose"], "growth_media": ["YPD"]},
            "YPG": {"carbon_source": ["glycerol"], "growth_media": ["YPG"]},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "repositories": {
                    "BrentLab/test": {
                        "dataset": {
                            "test": {"carbon_source": {"path": "media.carbon_source"}}
                        }
                    }
                }
            }
            yaml.dump(config, f)
            config_path = f.name

        try:
            vdb = VirtualDB(config_path)
            result = vdb._add_field_metadata(df, field_metadata)

            assert "carbon_source" in result.columns
            assert "growth_media" in result.columns
            assert (
                result.loc[result["condition"] == "YPD", "carbon_source"].iloc[0]
                == "glucose"
            )
            assert (
                result.loc[result["condition"] == "YPG", "carbon_source"].iloc[0]
                == "glycerol"
            )
        finally:
            Path(config_path).unlink()


class TestQuery:
    """Tests for query method - requires mocking HfQueryAPI."""

    def test_query_empty_result(self):
        """Test query with no matching datasets."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "repositories": {
                    "BrentLab/test": {
                        "dataset": {
                            "test": {"carbon_source": {"path": "media.carbon_source"}}
                        }
                    }
                }
            }
            yaml.dump(config, f)
            config_path = f.name

        try:
            vdb = VirtualDB(config_path)
            # Query with non-configured dataset should return empty
            result = vdb.query(datasets=[("BrentLab/other", "other")])
            assert isinstance(result, pd.DataFrame)
            assert result.empty
        finally:
            Path(config_path).unlink()


class TestComparativeDatasets:
    """Tests for comparative dataset field-based joins."""

    def test_parse_composite_identifier(self):
        """Test parsing composite identifiers."""
        composite_id = "BrentLab/harbison_2004;harbison_2004;sample_42"
        repo, config, sample = VirtualDB._parse_composite_identifier(composite_id)
        assert repo == "BrentLab/harbison_2004"
        assert config == "harbison_2004"
        assert sample == "sample_42"

    def test_parse_composite_identifier_invalid(self):
        """Test that invalid composite IDs raise errors."""
        with pytest.raises(ValueError, match="Invalid composite ID format"):
            VirtualDB._parse_composite_identifier("invalid:format")

    def test_get_comparative_fields_for_dataset(self):
        """Test getting comparative fields mapping."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "repositories": {
                    "BrentLab/primary": {
                        "dataset": {
                            "primary_data": {
                                "sample_id": {"field": "sample_id"},
                                "comparative_analyses": [
                                    {
                                        "repo": "BrentLab/comparative",
                                        "dataset": "comp_data",
                                        "via_field": "binding_id",
                                    }
                                ],
                            }
                        }
                    },
                    "BrentLab/comparative": {
                        "dataset": {
                            "comp_data": {
                                "dto_fdr": {"field": "dto_fdr"},
                                "dto_pvalue": {"field": "dto_empirical_pvalue"},
                            }
                        }
                    },
                }
            }
            yaml.dump(config, f)
            config_path = f.name

        try:
            vdb = VirtualDB(config_path)
            field_mapping = vdb._get_comparative_fields_for_dataset(
                "BrentLab/primary", "primary_data"
            )

            # Should have dto_fdr and dto_pvalue, but NOT binding_id (via_field)
            assert "dto_fdr" in field_mapping
            assert "dto_pvalue" in field_mapping
            assert "binding_id" not in field_mapping

            # Check mapping structure
            assert field_mapping["dto_fdr"]["comp_repo"] == "BrentLab/comparative"
            assert field_mapping["dto_fdr"]["comp_dataset"] == "comp_data"
            assert field_mapping["dto_fdr"]["via_field"] == "binding_id"
        finally:
            Path(config_path).unlink()

    def test_get_comparative_fields_no_links(self):
        """Test that datasets without comparative links return empty mapping."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "repositories": {
                    "BrentLab/primary": {
                        "dataset": {
                            "primary_data": {"sample_id": {"field": "sample_id"}}
                        }
                    }
                }
            }
            yaml.dump(config, f)
            config_path = f.name

        try:
            vdb = VirtualDB(config_path)
            field_mapping = vdb._get_comparative_fields_for_dataset(
                "BrentLab/primary", "primary_data"
            )
            assert field_mapping == {}
        finally:
            Path(config_path).unlink()

    def test_get_comparative_analyses(self):
        """Test getting comparative analysis relationships."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "repositories": {
                    "BrentLab/primary": {
                        "dataset": {
                            "primary_data": {
                                "sample_id": {"field": "sample_id"},
                                "comparative_analyses": [
                                    {
                                        "repo": "BrentLab/comparative",
                                        "dataset": "comp_data",
                                        "via_field": "binding_id",
                                    }
                                ],
                            }
                        }
                    },
                    "BrentLab/comparative": {
                        "dataset": {"comp_data": {"dto_fdr": {"field": "dto_fdr"}}}
                    },
                }
            }
            yaml.dump(config, f)
            config_path = f.name

        try:
            vdb = VirtualDB(config_path)
            info = vdb.get_comparative_analyses()

            # Check primary to comparative mapping
            assert "BrentLab/primary/primary_data" in info["primary_to_comparative"]
            links = info["primary_to_comparative"]["BrentLab/primary/primary_data"]
            assert len(links) == 1
            assert links[0]["comparative_repo"] == "BrentLab/comparative"
            assert links[0]["comparative_dataset"] == "comp_data"
            assert links[0]["via_field"] == "binding_id"

            # Check comparative fields
            assert "BrentLab/comparative/comp_data" in info["comparative_fields"]
            assert (
                "dto_fdr"
                in info["comparative_fields"]["BrentLab/comparative/comp_data"]
            )
        finally:
            Path(config_path).unlink()

    def test_get_comparative_analyses_filtered(self):
        """Test filtering comparative analyses by repo and config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "repositories": {
                    "BrentLab/primary1": {
                        "dataset": {
                            "data1": {
                                "sample_id": {"field": "sample_id"},
                                "comparative_analyses": [
                                    {
                                        "repo": "BrentLab/comp",
                                        "dataset": "comp_data",
                                        "via_field": "id1",
                                    }
                                ],
                            }
                        }
                    },
                    "BrentLab/primary2": {
                        "dataset": {
                            "data2": {
                                "sample_id": {"field": "sample_id"},
                                "comparative_analyses": [
                                    {
                                        "repo": "BrentLab/comp",
                                        "dataset": "comp_data",
                                        "via_field": "id2",
                                    }
                                ],
                            }
                        }
                    },
                }
            }
            yaml.dump(config, f)
            config_path = f.name

        try:
            vdb = VirtualDB(config_path)

            # Get all
            all_info = vdb.get_comparative_analyses()
            assert len(all_info["primary_to_comparative"]) == 2

            # Filter by repo and config
            filtered = vdb.get_comparative_analyses("BrentLab/primary1", "data1")
            assert len(filtered["primary_to_comparative"]) == 1
            assert "BrentLab/primary1/data1" in filtered["primary_to_comparative"]

            # Filter by repo only
            repo_filtered = vdb.get_comparative_analyses("BrentLab/primary2")
            assert len(repo_filtered["primary_to_comparative"]) == 1
            assert "BrentLab/primary2/data2" in repo_filtered["primary_to_comparative"]
        finally:
            Path(config_path).unlink()


# Note: Full integration tests with real HuggingFace datasets would go here
# but are excluded as they require network access and specific test datasets.
# These tests cover the core logic and would be supplemented with integration
# tests using the actual sample config and real datasets like harbison_2004.
