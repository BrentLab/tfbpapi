"""Tests for DatasetFilterResolver."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import pytest
import yaml

from tfbpapi.datainfo.filter_resolver import (
    DatasetFilterResolver,
    extract_compound_names,
    get_nested_value,
)


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_nested_value_simple(self):
        """Test getting nested values with dot notation."""
        data = {
            "environmental_conditions": {
                "media": {
                    "name": "YPD",
                    "carbon_source": [{"compound": "D-glucose"}]
                },
                "temperature_celsius": 30
            }
        }

        assert get_nested_value(data, "environmental_conditions.temperature_celsius") == 30
        assert get_nested_value(data, "environmental_conditions.media.name") == "YPD"
        assert get_nested_value(data, "nonexistent.path") is None
        assert get_nested_value(data, "environmental_conditions.media.nonexistent") is None

    def test_get_nested_value_fallback(self):
        """Test fallback path resolution for field-level definitions."""
        # Field-level definition (no environmental_conditions wrapper)
        field_def = {
            "media": {
                "name": "YPD",
                "carbon_source": [{"compound": "D-glucose"}]
            },
            "temperature_celsius": 30
        }

        # Path with environmental_conditions should fallback to path without it
        assert get_nested_value(field_def, "environmental_conditions.media.name") == "YPD"
        assert get_nested_value(field_def, "environmental_conditions.temperature_celsius") == 30

        # Direct path should still work
        assert get_nested_value(field_def, "media.name") == "YPD"
        assert get_nested_value(field_def, "temperature_celsius") == 30

    def test_extract_compound_names(self):
        """Test extracting compound names from various formats."""
        # List of dicts
        value1 = [
            {"compound": "D-glucose", "concentration_percent": 2},
            {"compound": "D-galactose", "concentration_percent": 1}
        ]
        assert extract_compound_names(value1) == ["D-glucose", "D-galactose"]

        # String
        assert extract_compound_names("D-glucose") == ["D-glucose"]

        # None
        assert extract_compound_names(None) == []

        # "unspecified"
        assert extract_compound_names("unspecified") == []

        # Empty list
        assert extract_compound_names([]) == []


class TestDatasetFilterResolver:
    """Test DatasetFilterResolver class."""

    @pytest.fixture
    def simple_config(self):
        """Create a simple test configuration."""
        return {
            "filters": {
                "carbon_source": ["D-glucose", "D-galactose"],
                "temperature_celsius": [30]
            },
            "dataset_mappings": {
                "BrentLab/harbison_2004": {
                    "datasets": {
                        "harbison_2004": {
                            "carbon_source": {
                                "field": "condition",
                                "path": "media.carbon_source"
                            },
                            "temperature_celsius": {
                                "field": "condition",
                                "path": "temperature_celsius"
                            }
                        }
                    }
                }
            }
        }

    @pytest.fixture
    def config_file(self, simple_config, tmp_path):
        """Create a temporary config file."""
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(simple_config, f)
        return config_path

    def test_init(self, config_file):
        """Test initialization."""
        resolver = DatasetFilterResolver(config_file)

        assert len(resolver.filters) == 2
        assert "carbon_source" in resolver.filters
        assert resolver.filters["carbon_source"] == ["D-glucose", "D-galactose"]
        assert len(resolver.mappings) == 1
        assert "BrentLab/harbison_2004" in resolver.mappings

    def test_init_missing_file(self):
        """Test initialization with missing config file."""
        with pytest.raises(FileNotFoundError):
            DatasetFilterResolver("nonexistent.yaml")

    def test_resolve_filters_mode_conditions(self, config_file):
        """Test resolve_filters in conditions mode."""
        resolver = DatasetFilterResolver(config_file)

        # This will actually try to load the DataCard, so it's more of an integration test
        # For now, test the structure
        results = resolver.resolve_filters(
            repos=[("BrentLab/harbison_2004", "harbison_2004")],
            mode="conditions"
        )

        assert "BrentLab/harbison_2004" in results
        result = results["BrentLab/harbison_2004"]

        # Should have included field
        assert "included" in result

        # If included, should have matching_field_values
        if result["included"]:
            assert "matching_field_values" in result

    def test_resolve_filters_invalid_mode(self, config_file):
        """Test resolve_filters with invalid mode."""
        resolver = DatasetFilterResolver(config_file)

        with pytest.raises(ValueError, match="Invalid mode"):
            resolver.resolve_filters(
                repos=[("BrentLab/harbison_2004", "harbison_2004")],
                mode="invalid"
            )

    def test_repr(self, config_file):
        """Test string representation."""
        resolver = DatasetFilterResolver(config_file)
        repr_str = repr(resolver)

        assert "DatasetFilterResolver" in repr_str
        assert "2 filters" in repr_str
        assert "1 datasets" in repr_str

    def test_hierarchical_config(self, tmp_path):
        """Test hierarchical configuration with repo_level and dataset-specific mappings."""
        config = {
            "filters": {
                "carbon_source": ["D-glucose"],
                "temperature_celsius": [30]
            },
            "dataset_mappings": {
                "BrentLab/test_repo": {
                    "repo_level": {
                        "carbon_source": {
                            "path": "environmental_conditions.media.carbon_source"
                        }
                    },
                    "datasets": {
                        "dataset1": {
                            "temperature_celsius": {
                                "field": "condition",
                            "path": "environmental_conditions.temperature_celsius"
                            }
                        },
                        "dataset2": {
                            "temperature_celsius": {
                                "field": "condition",
                            "path": "custom.temp.path"
                            }
                        }
                    }
                }
            }
        }

        config_path = tmp_path / "hierarchical_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        resolver = DatasetFilterResolver(config_path)

        # Test property mapping resolution
        mappings1 = resolver._get_property_mappings("BrentLab/test_repo", "dataset1")
        assert "carbon_source" in mappings1  # From repo_level
        assert "temperature_celsius" in mappings1  # From dataset-specific
        assert mappings1["temperature_celsius"]["path"] == "environmental_conditions.temperature_celsius"

        mappings2 = resolver._get_property_mappings("BrentLab/test_repo", "dataset2")
        assert "carbon_source" in mappings2  # From repo_level
        assert "temperature_celsius" in mappings2  # Overridden by dataset-specific
        assert mappings2["temperature_celsius"]["path"] == "custom.temp.path"


class TestRealDataCards:
    """Integration tests with real datacards."""

    def test_harbison_2004_glucose_filter(self, tmp_path):
        """Test filtering harbison_2004 for glucose samples."""
        # Create config for glucose filtering
        config = {
            "filters": {
                "carbon_source": ["D-glucose"]
            },
            "dataset_mappings": {
                "BrentLab/harbison_2004": {
                    "datasets": {
                        "harbison_2004": {
                            "carbon_source": {
                                "field": "condition",
                            "path": "media.carbon_source"
                            }
                        }
                    }
                }
            }
        }

        config_path = tmp_path / "glucose_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        resolver = DatasetFilterResolver(config_path)
        results = resolver.resolve_filters(
            repos=[("BrentLab/harbison_2004", "harbison_2004")],
            mode="conditions"
        )

        assert "BrentLab/harbison_2004" in results
        result = results["BrentLab/harbison_2004"]

        # Should be included
        assert result["included"] is True

        # Should have matching field values
        assert "matching_field_values" in result

        # Should have condition field with some matching values
        if "condition" in result["matching_field_values"]:
            matching = result["matching_field_values"]["condition"]
            # YPD, HEAT, H2O2Hi, H2O2Lo, Acid, Alpha, BUT14, BUT90 all have D-glucose
            expected_glucose_conditions = ["YPD", "HEAT", "H2O2Hi", "H2O2Lo", "Acid", "Alpha", "BUT14", "BUT90"]
            for cond in expected_glucose_conditions:
                assert cond in matching, f"{cond} should be in matching conditions"

    def test_harbison_2004_galactose_filter(self, tmp_path):
        """Test filtering harbison_2004 for galactose samples."""
        config = {
            "filters": {
                "carbon_source": ["D-galactose"]
            },
            "dataset_mappings": {
                "BrentLab/harbison_2004": {
                    "datasets": {
                        "harbison_2004": {
                            "carbon_source": {
                                "field": "condition",
                            "path": "media.carbon_source"
                            }
                        }
                    }
                }
            }
        }

        config_path = tmp_path / "galactose_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        resolver = DatasetFilterResolver(config_path)
        results = resolver.resolve_filters(
            repos=[("BrentLab/harbison_2004", "harbison_2004")],
            mode="conditions"
        )

        result = results["BrentLab/harbison_2004"]
        assert result["included"] is True

        # GAL condition has D-galactose
        if "condition" in result["matching_field_values"]:
            matching = result["matching_field_values"]["condition"]
            assert "GAL" in matching

    def test_harbison_2004_samples_mode(self, tmp_path):
        """Test retrieving sample-level metadata."""
        config = {
            "filters": {
                "carbon_source": ["D-glucose"]
            },
            "dataset_mappings": {
                "BrentLab/harbison_2004": {
                    "datasets": {
                        "harbison_2004": {
                            "carbon_source": {
                                "field": "condition",
                            "path": "media.carbon_source"
                            }
                        }
                    }
                }
            }
        }

        config_path = tmp_path / "samples_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        resolver = DatasetFilterResolver(config_path)
        results = resolver.resolve_filters(
            repos=[("BrentLab/harbison_2004", "harbison_2004")],
            mode="samples"
        )

        result = results["BrentLab/harbison_2004"]
        assert result["included"] is True
        assert "data" in result

        # Should have a DataFrame
        df = result["data"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0  # Should have some samples

    def test_harbison_2004_full_data_mode(self, tmp_path):
        """Test retrieving full data."""
        config = {
            "filters": {
                "carbon_source": ["D-glucose"]
            },
            "dataset_mappings": {
                "BrentLab/harbison_2004": {
                    "datasets": {
                        "harbison_2004": {
                            "carbon_source": {
                                "field": "condition",
                            "path": "media.carbon_source"
                            }
                        }
                    }
                }
            }
        }

        config_path = tmp_path / "full_data_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        resolver = DatasetFilterResolver(config_path)
        results = resolver.resolve_filters(
            repos=[("BrentLab/harbison_2004", "harbison_2004")],
            mode="full_data"
        )

        result = results["BrentLab/harbison_2004"]
        assert result["included"] is True
        assert "data" in result

        # Should have a DataFrame
        df = result["data"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0  # Should have data rows
