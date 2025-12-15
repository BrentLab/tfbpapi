"""Tests for DatasetFilterResolver."""

import pandas as pd
import pytest

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
                "media": {"name": "YPD", "carbon_source": [{"compound": "D-glucose"}]},
                "temperature_celsius": 30,
            }
        }

        assert (
            get_nested_value(data, "environmental_conditions.temperature_celsius") == 30
        )
        assert get_nested_value(data, "environmental_conditions.media.name") == "YPD"
        assert get_nested_value(data, "nonexistent.path") is None
        assert (
            get_nested_value(data, "environmental_conditions.media.nonexistent") is None
        )

    def test_get_nested_value_no_fallback(self):
        """Test that paths must be correct - no automatic fallback."""
        # Field-level definition (no experimental_conditions wrapper)
        field_def = {
            "media": {"name": "YPD", "carbon_source": [{"compound": "D-glucose"}]},
            "temperature_celsius": 30,
        }

        # Wrong path should return None (no fallback)
        assert get_nested_value(field_def, "experimental_conditions.media.name") is None
        assert (
            get_nested_value(field_def, "experimental_conditions.temperature_celsius")
            is None
        )

        # Correct direct paths work
        assert get_nested_value(field_def, "media.name") == "YPD"
        assert get_nested_value(field_def, "temperature_celsius") == 30

        # Repo-level definition (with experimental_conditions wrapper)
        repo_def = {
            "experimental_conditions": {
                "media": {"name": "SC", "carbon_source": [{"compound": "D-glucose"}]},
                "temperature_celsius": 30,
            }
        }

        # Full path works
        assert get_nested_value(repo_def, "experimental_conditions.media.name") == "SC"
        assert (
            get_nested_value(repo_def, "experimental_conditions.temperature_celsius")
            == 30
        )

        # Shortened path does NOT work (no fallback)
        assert get_nested_value(repo_def, "media.name") is None
        assert get_nested_value(repo_def, "temperature_celsius") is None

    def test_extract_compound_names(self):
        """Test extracting compound names from various formats."""
        # List of dicts
        value1 = [
            {"compound": "D-glucose", "concentration_percent": 2},
            {"compound": "D-galactose", "concentration_percent": 1},
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

    def test_init(self, write_config):
        """Test initialization."""
        config = {
            "filters": {
                "carbon_source": ["D-glucose", "D-galactose"],
                "temperature_celsius": [30],
            },
            "BrentLab/harbison_2004": {
                "dataset": {
                    "harbison_2004": {
                        "carbon_source": {
                            "field": "condition",
                            "path": "media.carbon_source",
                        },
                        "temperature_celsius": {
                            "field": "condition",
                            "path": "temperature_celsius",
                        },
                    }
                }
            },
        }
        config_file = write_config(config)
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

    def test_resolve_filters_mode_conditions(self, write_config):
        """Test resolve_filters in conditions mode."""
        config = {
            "filters": {
                "carbon_source": ["D-glucose", "D-galactose"],
                "temperature_celsius": [30],
            },
            "BrentLab/harbison_2004": {
                "dataset": {
                    "harbison_2004": {
                        "carbon_source": {
                            "field": "condition",
                            "path": "media.carbon_source",
                        },
                        "temperature_celsius": {
                            "field": "condition",
                            "path": "temperature_celsius",
                        },
                    }
                }
            },
        }
        config_file = write_config(config)
        resolver = DatasetFilterResolver(config_file)

        # This will actually try to load the DataCard,
        # so it's more of an integration test For now, test the structure
        results = resolver.resolve_filters(
            repos=[("BrentLab/harbison_2004", "harbison_2004")], mode="conditions"
        )

        assert "BrentLab/harbison_2004" in results
        result = results["BrentLab/harbison_2004"]

        # Should have included field
        assert "included" in result

        # If included, should have matching_field_values
        if result["included"]:
            assert "matching_field_values" in result

    def test_resolve_filters_invalid_mode(self, write_config):
        """Test resolve_filters with invalid mode."""
        config = {
            "filters": {
                "carbon_source": ["D-glucose", "D-galactose"],
                "temperature_celsius": [30],
            },
            "BrentLab/harbison_2004": {
                "dataset": {
                    "harbison_2004": {
                        "carbon_source": {
                            "field": "condition",
                            "path": "media.carbon_source",
                        },
                        "temperature_celsius": {
                            "field": "condition",
                            "path": "temperature_celsius",
                        },
                    }
                }
            },
        }
        config_file = write_config(config)
        resolver = DatasetFilterResolver(config_file)

        with pytest.raises(ValueError, match="Invalid mode"):
            resolver.resolve_filters(
                repos=[("BrentLab/harbison_2004", "harbison_2004")], mode="invalid"
            )

    def test_repr(self, write_config):
        """Test string representation."""
        config = {
            "filters": {
                "carbon_source": ["D-glucose", "D-galactose"],
                "temperature_celsius": [30],
            },
            "BrentLab/harbison_2004": {
                "dataset": {
                    "harbison_2004": {
                        "carbon_source": {
                            "field": "condition",
                            "path": "media.carbon_source",
                        },
                        "temperature_celsius": {
                            "field": "condition",
                            "path": "temperature_celsius",
                        },
                    }
                }
            },
        }
        config_file = write_config(config)
        resolver = DatasetFilterResolver(config_file)
        repr_str = repr(resolver)

        assert "DatasetFilterResolver" in repr_str
        assert "2 filters" in repr_str
        assert "1 datasets" in repr_str


class TestRealDataCards:
    """Integration tests with real datacards."""

    def test_harbison_2004_glucose_filter(self, harbison_2004_datacard, write_config):
        """Test filtering harbison_2004 for glucose samples."""
        config = {
            "filters": {"carbon_source": ["D-glucose"]},
            "BrentLab/harbison_2004": {
                "dataset": {
                    "harbison_2004": {
                        "carbon_source": {
                            "field": "condition",
                            "path": "media.carbon_source",
                        }
                    }
                }
            },
        }
        config_path = write_config(config)
        resolver = DatasetFilterResolver(config_path)
        results = resolver.resolve_filters(
            repos=[("BrentLab/harbison_2004", "harbison_2004")], mode="conditions"
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
            expected_glucose_conditions = [
                "YPD",
                "HEAT",
                "H2O2Hi",
                "H2O2Lo",
                "Acid",
                "Alpha",
                "BUT14",
                "BUT90",
            ]
            for cond in expected_glucose_conditions:
                assert cond in matching, f"{cond} should be in matching conditions"

    def test_harbison_2004_galactose_filter(self, harbison_2004_datacard, write_config):
        """Test filtering harbison_2004 for galactose samples."""
        config = {
            "filters": {"carbon_source": ["D-galactose"]},
            "BrentLab/harbison_2004": {
                "dataset": {
                    "harbison_2004": {
                        "carbon_source": {
                            "field": "condition",
                            "path": "media.carbon_source",
                        }
                    }
                }
            },
        }
        config_path = write_config(config)
        resolver = DatasetFilterResolver(config_path)
        results = resolver.resolve_filters(
            repos=[("BrentLab/harbison_2004", "harbison_2004")], mode="conditions"
        )

        result = results["BrentLab/harbison_2004"]
        assert result["included"] is True

        # GAL condition has D-galactose
        if "condition" in result["matching_field_values"]:
            matching = result["matching_field_values"]["condition"]
            assert "GAL" in matching

    def test_harbison_2004_samples_mode(self, harbison_2004_datacard, write_config):
        """Test retrieving sample-level metadata."""
        config = {
            "filters": {"carbon_source": ["D-glucose"]},
            "BrentLab/harbison_2004": {
                "dataset": {
                    "harbison_2004": {
                        "carbon_source": {
                            "field": "condition",
                            "path": "media.carbon_source",
                        }
                    }
                }
            },
        }
        config_path = write_config(config)
        resolver = DatasetFilterResolver(config_path)
        results = resolver.resolve_filters(
            repos=[("BrentLab/harbison_2004", "harbison_2004")], mode="samples"
        )

        result = results["BrentLab/harbison_2004"]
        assert result["included"] is True
        assert "data" in result

        # Should have a DataFrame
        df = result["data"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0  # Should have some samples

    def test_harbison_2004_full_data_mode(self, harbison_2004_datacard, write_config):
        """Test retrieving full data."""
        config = {
            "filters": {"carbon_source": ["D-glucose"]},
            "BrentLab/harbison_2004": {
                "dataset": {
                    "harbison_2004": {
                        "carbon_source": {
                            "field": "condition",
                            "path": "media.carbon_source",
                        }
                    }
                }
            },
        }
        config_path = write_config(config)
        resolver = DatasetFilterResolver(config_path)
        results = resolver.resolve_filters(
            repos=[("BrentLab/harbison_2004", "harbison_2004")], mode="full_data"
        )

        result = results["BrentLab/harbison_2004"]
        assert result["included"] is True
        assert "data" in result

        # Should have a DataFrame
        df = result["data"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0  # Should have data rows

    def test_hackett_2020_glucose_filter(self, hackett_2020_datacard, write_config):
        """Test filtering hackett_2020 for glucose samples."""
        config = {
            "filters": {"carbon_source": ["D-glucose"]},
            "BrentLab/hackett_2020": {
                "dataset": {
                    "hackett_2020": {"carbon_source": {"path": "media.carbon_source"}}
                }
            },
        }
        config_path = write_config(config)
        resolver = DatasetFilterResolver(config_path)
        results = resolver.resolve_filters(
            repos=[("BrentLab/hackett_2020", "hackett_2020")], mode="conditions"
        )

        assert "BrentLab/hackett_2020" in results
        result = results["BrentLab/hackett_2020"]

        # Should be included (hackett_2020 has D-glucose at repo level)
        assert result["included"] is True

    def test_kemmeren_2014_glucose_filter(self, kemmeren_2014_datacard, write_config):
        """Test filtering kemmeren_2014 for glucose samples."""
        config = {
            "filters": {"carbon_source": ["D-glucose"]},
            "BrentLab/kemmeren_2014": {
                "dataset": {
                    "kemmeren_2014": {"carbon_source": {"path": "media.carbon_source"}}
                }
            },
        }
        config_path = write_config(config)
        resolver = DatasetFilterResolver(config_path)
        results = resolver.resolve_filters(
            repos=[("BrentLab/kemmeren_2014", "kemmeren_2014")], mode="conditions"
        )

        assert "BrentLab/kemmeren_2014" in results
        result = results["BrentLab/kemmeren_2014"]

        # Should be included (kemmeren_2014 has D-glucose at repo level)
        assert result["included"] is True

    def test_kemmeren_2014_temperature_filter(
        self, kemmeren_2014_datacard, write_config
    ):
        """Test filtering kemmeren_2014 for temperature."""
        config = {
            "filters": {"temperature_celsius": [30]},
            "BrentLab/kemmeren_2014": {
                "dataset": {
                    "kemmeren_2014": {
                        "temperature_celsius": {"path": "temperature_celsius"}
                    }
                }
            },
        }
        config_path = write_config(config)
        resolver = DatasetFilterResolver(config_path)
        results = resolver.resolve_filters(
            repos=[("BrentLab/kemmeren_2014", "kemmeren_2014")], mode="conditions"
        )

        assert "BrentLab/kemmeren_2014" in results
        result = results["BrentLab/kemmeren_2014"]

        # Should be included (kemmeren_2014 has temperature_celsius: 30 at repo level)
        assert result["included"] is True

    def test_multi_repo_glucose_filter(
        self,
        harbison_2004_datacard,
        hackett_2020_datacard,
        kemmeren_2014_datacard,
        write_config,
    ):
        """Test filtering D-glucose across multiple repos."""
        config = {
            "filters": {"carbon_source": ["D-glucose"]},
            "BrentLab/harbison_2004": {
                "dataset": {
                    "harbison_2004": {
                        "carbon_source": {
                            "field": "condition",
                            "path": "media.carbon_source",
                        }
                    }
                }
            },
            "BrentLab/hackett_2020": {
                "dataset": {
                    "hackett_2020": {"carbon_source": {"path": "media.carbon_source"}}
                }
            },
            "BrentLab/kemmeren_2014": {
                "dataset": {
                    "kemmeren_2014": {"carbon_source": {"path": "media.carbon_source"}}
                }
            },
        }
        config_path = write_config(config)
        resolver = DatasetFilterResolver(config_path)
        results = resolver.resolve_filters(
            repos=[
                ("BrentLab/harbison_2004", "harbison_2004"),
                ("BrentLab/hackett_2020", "hackett_2020"),
                ("BrentLab/kemmeren_2014", "kemmeren_2014"),
            ],
            mode="conditions",
        )

        # All three repos should be included
        assert "BrentLab/harbison_2004" in results
        assert results["BrentLab/harbison_2004"]["included"] is True

        assert "BrentLab/hackett_2020" in results
        assert results["BrentLab/hackett_2020"]["included"] is True

        assert "BrentLab/kemmeren_2014" in results
        assert results["BrentLab/kemmeren_2014"]["included"] is True
