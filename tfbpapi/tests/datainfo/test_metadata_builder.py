"""
Tests for MetadataBuilder.

Tests normalization logic, metadata extraction, and builder functionality.

"""

from pathlib import Path

import pytest
import yaml

from tfbpapi.datainfo.metadata_builder import (
    MetadataBuilder,
    get_nested_value,
    normalize_value,
)


class TestGetNestedValue:
    """Test get_nested_value function."""

    def test_simple_path(self):
        """Test simple one-level path."""
        data = {"temperature": 30}
        assert get_nested_value(data, "temperature") == 30

    def test_nested_path(self):
        """Test multi-level nested path."""
        data = {"media": {"carbon_source": "D-glucose"}}
        assert get_nested_value(data, "media.carbon_source") == "D-glucose"

    def test_deeply_nested_path(self):
        """Test deeply nested path."""
        data = {"level1": {"level2": {"level3": {"value": 42}}}}
        assert get_nested_value(data, "level1.level2.level3.value") == 42

    def test_missing_key(self):
        """Test that missing keys return None."""
        data = {"temperature": 30}
        assert get_nested_value(data, "pressure") is None

    def test_missing_intermediate_key(self):
        """Test that missing intermediate keys return None."""
        data = {"media": {"temperature": 30}}
        assert get_nested_value(data, "media.carbon_source.compound") is None

    def test_non_dict_input(self):
        """Test that non-dict input returns None."""
        assert get_nested_value("not a dict", "path") is None
        assert get_nested_value(None, "path") is None


class TestGetNestedValueListExtraction:
    """Test get_nested_value list extraction functionality."""

    def test_extract_from_list_of_dicts(self):
        """Test extracting property from list of dicts."""
        data = {
            "media": {
                "carbon_source": [
                    {"compound": "D-glucose", "concentration_percent": 2},
                    {"compound": "D-galactose", "concentration_percent": 1},
                ]
            }
        }
        result = get_nested_value(data, "media.carbon_source.compound")
        assert result == ["D-glucose", "D-galactose"]

    def test_extract_concentration_from_list(self):
        """Test extracting numeric property from list of dicts."""
        data = {
            "media": {
                "carbon_source": [
                    {"compound": "D-glucose", "concentration_percent": 2},
                    {"compound": "D-galactose", "concentration_percent": 1},
                ]
            }
        }
        result = get_nested_value(data, "media.carbon_source.concentration_percent")
        assert result == [2, 1]

    def test_get_list_itself(self):
        """Test getting the list without extracting a property."""
        data = {
            "media": {
                "carbon_source": [
                    {"compound": "D-glucose"},
                    {"compound": "D-galactose"},
                ]
            }
        }
        result = get_nested_value(data, "media.carbon_source")
        expected = [
            {"compound": "D-glucose"},
            {"compound": "D-galactose"},
        ]
        assert result == expected

    def test_extract_from_single_item_list(self):
        """Test extracting from list with single item."""
        data = {
            "media": {
                "carbon_source": [{"compound": "D-glucose"}]
            }
        }
        result = get_nested_value(data, "media.carbon_source.compound")
        assert result == ["D-glucose"]

    def test_extract_missing_property_from_list(self):
        """Test extracting non-existent property from list items."""
        data = {
            "media": {
                "carbon_source": [
                    {"compound": "D-glucose"},
                    {"compound": "D-galactose"},
                ]
            }
        }
        result = get_nested_value(data, "media.carbon_source.missing_key")
        assert result is None

    def test_nested_list_extraction(self):
        """Test extracting from nested structures with lists."""
        data = {
            "level1": {
                "level2": [
                    {"level3": {"value": "a"}},
                    {"level3": {"value": "b"}},
                ]
            }
        }
        result = get_nested_value(data, "level1.level2.level3.value")
        assert result == ["a", "b"]


class TestNormalizeValue:
    """Test normalize_value function."""

    def test_normalize_with_alias_match(self):
        """Test normalization when value matches an alias."""
        aliases = {"glucose": ["D-glucose", "dextrose"]}
        assert normalize_value("D-glucose", aliases) == "glucose"
        assert normalize_value("dextrose", aliases) == "glucose"

    def test_normalize_case_insensitive(self):
        """Test that matching is case-insensitive."""
        aliases = {"glucose": ["D-glucose"]}
        assert normalize_value("d-glucose", aliases) == "glucose"
        assert normalize_value("D-GLUCOSE", aliases) == "glucose"
        assert normalize_value("D-Glucose", aliases) == "glucose"

    def test_normalize_no_match_passthrough(self):
        """Test pass-through when no alias matches."""
        aliases = {"glucose": ["D-glucose"]}
        assert normalize_value("maltose", aliases) == "maltose"
        assert normalize_value("galactose", aliases) == "galactose"

    def test_normalize_no_aliases(self):
        """Test pass-through when no aliases provided."""
        assert normalize_value("D-glucose", None) == "D-glucose"
        assert normalize_value("maltose", None) == "maltose"

    def test_normalize_empty_aliases(self):
        """Test pass-through when empty aliases dict."""
        assert normalize_value("D-glucose", {}) == "D-glucose"

    def test_normalize_numeric_values(self):
        """Test normalization with numeric actual values."""
        aliases = {"thirty": [30, "30"]}
        assert normalize_value(30, aliases) == "thirty"
        assert normalize_value("30", aliases) == "thirty"

    def test_normalize_numeric_passthrough(self):
        """Test that unmatched numeric values pass through as strings."""
        aliases = {"thirty": [30]}
        assert normalize_value(37, aliases) == "37"

    def test_normalize_boolean_values(self):
        """Test normalization with boolean values."""
        aliases = {"present": [True, "true", "True"]}
        assert normalize_value(True, aliases) == "present"
        # Note: bool to str conversion makes this "True"
        assert normalize_value("true", aliases) == "present"

    def test_normalize_multiple_aliases(self):
        """Test with multiple alias mappings."""
        aliases = {
            "glucose": ["D-glucose", "dextrose"],
            "galactose": ["D-galactose", "Galactose"],
        }
        assert normalize_value("D-glucose", aliases) == "glucose"
        assert normalize_value("DEXTROSE", aliases) == "glucose"
        assert normalize_value("d-galactose", aliases) == "galactose"
        assert normalize_value("Galactose", aliases) == "galactose"
        # No match
        assert normalize_value("maltose", aliases) == "maltose"


@pytest.fixture
def write_config(tmp_path):
    """Fixture to write YAML config files."""

    def _write(config_data):
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        return config_path

    return _write


class TestMetadataBuilder:
    """Test MetadataBuilder class."""

    def test_init_with_aliases(self, write_config):
        """Test initialization with factor aliases."""
        config = {
            "factor_aliases": {"carbon_source": {"glucose": ["D-glucose", "dextrose"]}},
            "BrentLab/test": {
                "dataset": {"test": {"carbon_source": {"path": "media.carbon_source"}}}
            },
        }
        config_file = write_config(config)
        builder = MetadataBuilder(config_file)

        assert len(builder.factor_aliases) == 1
        assert "carbon_source" in builder.factor_aliases
        assert "glucose" in builder.factor_aliases["carbon_source"]

    def test_init_without_aliases(self, write_config):
        """Test initialization without factor aliases."""
        config = {
            "BrentLab/test": {
                "dataset": {"test": {"carbon_source": {"path": "media.carbon_source"}}}
            }
        }
        config_file = write_config(config)
        builder = MetadataBuilder(config_file)

        assert builder.factor_aliases == {}

    def test_invalid_mode(self, write_config):
        """Test that invalid mode raises ValueError."""
        config = {
            "BrentLab/test": {
                "dataset": {"test": {"carbon_source": {"path": "media.carbon_source"}}}
            }
        }
        config_file = write_config(config)
        builder = MetadataBuilder(config_file)

        with pytest.raises(ValueError) as exc_info:
            builder.build_metadata(
                repos=[("BrentLab/test", "test")], mode="invalid_mode"
            )
        assert "Invalid mode" in str(exc_info.value)

    def test_build_metadata_missing_repo_config(self, write_config):
        """Test handling of missing repository configuration."""
        config = {
            "BrentLab/test": {
                "dataset": {"test": {"carbon_source": {"path": "media.carbon_source"}}}
            }
        }
        config_file = write_config(config)
        builder = MetadataBuilder(config_file)

        results = builder.build_metadata(
            repos=[("BrentLab/missing", "dataset")], mode="conditions"
        )

        assert "BrentLab/missing" in results
        assert "error" in results["BrentLab/missing"]
        assert "No property mappings" in results["BrentLab/missing"]["error"]

    def test_repr(self, write_config):
        """Test string representation."""
        config = {
            "factor_aliases": {
                "carbon_source": {"glucose": ["D-glucose"]},
                "temperature": {"thirty": [30]},
            },
            "BrentLab/test1": {
                "dataset": {"test": {"carbon_source": {"path": "media.carbon_source"}}}
            },
            "BrentLab/test2": {
                "dataset": {"test": {"temperature": {"path": "temperature_celsius"}}}
            },
        }
        config_file = write_config(config)
        builder = MetadataBuilder(config_file)

        repr_str = repr(builder)
        assert "MetadataBuilder" in repr_str
        assert "2 properties" in repr_str
        assert "2 repositories" in repr_str

    def test_repr_no_aliases(self, write_config):
        """Test string representation with no aliases."""
        config = {
            "BrentLab/test": {
                "dataset": {"test": {"carbon_source": {"path": "media.carbon_source"}}}
            }
        }
        config_file = write_config(config)
        builder = MetadataBuilder(config_file)

        repr_str = repr(builder)
        assert "MetadataBuilder" in repr_str
        assert "0 properties" in repr_str
        assert "1 repositories" in repr_str


class TestMetadataBuilderIntegration:
    """Integration tests with real datacards (if available)."""

    def test_build_metadata_conditions_mode(self, write_config):
        """Test building metadata in conditions mode."""
        # This is a minimal test that doesn't require actual datacards
        # In practice, you'd use real datacards from HuggingFace
        config = {
            "factor_aliases": {"carbon_source": {"glucose": ["D-glucose"]}},
            "BrentLab/test": {
                "dataset": {
                    "test": {
                        "carbon_source": {
                            "field": "condition",
                            "path": "media.carbon_source",
                        }
                    }
                }
            },
        }
        config_file = write_config(config)
        builder = MetadataBuilder(config_file)

        # This would fail without a real datacard, but tests the structure
        # In actual usage, you'd mock DataCard or use real repos
        # For now, just verify the method exists and can be called
        assert hasattr(builder, "build_metadata")
        assert callable(builder.build_metadata)

    def test_get_property_mappings(self, write_config):
        """Test _get_property_mappings method."""
        config = {
            "BrentLab/test": {
                "temperature": {"path": "temperature_celsius"},
                "dataset": {
                    "test_dataset": {"carbon_source": {"path": "media.carbon_source"}}
                },
            }
        }
        config_file = write_config(config)
        builder = MetadataBuilder(config_file)

        mappings = builder._get_property_mappings("BrentLab/test", "test_dataset")

        assert "temperature" in mappings
        assert "carbon_source" in mappings
        assert mappings["temperature"].path == "temperature_celsius"
        assert mappings["carbon_source"].path == "media.carbon_source"

    def test_get_property_mappings_missing_repo(self, write_config):
        """Test _get_property_mappings with missing repository."""
        config = {
            "BrentLab/test": {
                "dataset": {"test": {"carbon_source": {"path": "media.carbon_source"}}}
            }
        }
        config_file = write_config(config)
        builder = MetadataBuilder(config_file)

        mappings = builder._get_property_mappings("BrentLab/missing", "dataset")
        assert mappings == {}
