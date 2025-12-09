"""Tests for MetadataManager."""

import pandas as pd
import pytest

from tfbpapi.datainfo import DataCard, MetadataManager
from tfbpapi.datainfo.models import FieldRole


class TestMetadataManagerBasic:
    """Basic tests for MetadataManager instantiation and structure."""

    def test_instantiation(self):
        """Test MetadataManager can be instantiated."""
        mgr = MetadataManager()
        assert mgr is not None
        assert hasattr(mgr, "_conn")
        assert hasattr(mgr, "_registered_datasets")
        assert hasattr(mgr, "_view_names")

    def test_instantiation_with_cache(self):
        """Test MetadataManager with cache enabled."""
        from pathlib import Path

        cache_dir = Path("/tmp/metadata_cache")
        mgr = MetadataManager(cache_dir=cache_dir, cache=True)
        assert mgr._cache_dir == cache_dir
        assert mgr._cache_enabled is True

    def test_get_active_configs_empty(self):
        """Test get_active_configs with no registered datasets."""
        mgr = MetadataManager()
        configs = mgr.get_active_configs()
        assert configs == []

    def test_get_summary_empty(self):
        """Test get_summary with no registered datasets."""
        mgr = MetadataManager()
        summary = mgr.get_summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 0


class TestDataCardExtractMetadataSchema:
    """Tests for DataCard.extract_metadata_schema method."""

    def test_extract_metadata_schema_structure(self):
        """Test that extract_metadata_schema returns correct structure."""
        # Note: This test uses a mock since we need a real datacard
        # In actual testing, we'd use a real HuggingFace dataset
        # For now, we just test the structure

        # Mock test - verify the method exists
        from tfbpapi.datainfo.datacard import DataCard

        assert hasattr(DataCard, "extract_metadata_schema")


class TestMetadataManagerHelpers:
    """Tests for MetadataManager helper methods."""

    def test_sanitize_view_name(self):
        """Test view name sanitization."""
        mgr = MetadataManager()
        view_name = mgr._sanitize_view_name("BrentLab/dataset-name", "config_name")
        assert view_name == "BrentLab_dataset_name_config_name_metadata"
        assert " " not in view_name
        assert "/" not in view_name
        assert "-" not in view_name

    def test_format_compound_simple(self):
        """Test formatting compound as simple string."""
        mgr = MetadataManager()
        result = mgr._format_compound("carbon_source", "D-glucose")
        assert result == "carbon_source:D-glucose"

    def test_format_compound_with_percent(self):
        """Test formatting compound with concentration percent."""
        mgr = MetadataManager()
        compound = {"name": "D-glucose", "concentration_percent": 2.0}
        result = mgr._format_compound("carbon_source", compound)
        assert result == "carbon_source:D-glucose@2.0%"

    def test_format_compound_with_grams(self):
        """Test formatting compound with g/L concentration."""
        mgr = MetadataManager()
        compound = {"name": "ammonium_sulfate", "concentration_g_per_l": 5.0}
        result = mgr._format_compound("nitrogen_source", compound)
        assert result == "nitrogen_source:ammonium_sulfate@5.0g/L"

    def test_flatten_condition_definition_empty(self):
        """Test flattening empty definition."""
        mgr = MetadataManager()
        result = mgr._flatten_condition_definition({})
        assert result["growth_media"] == "unspecified"
        assert result["components"] == ""

    def test_flatten_condition_definition_with_media(self):
        """Test flattening definition with media."""
        mgr = MetadataManager()
        definition = {
            "environmental_conditions": {
                "media": {
                    "name": "YPD",
                    "carbon_source": [
                        {"name": "D-glucose", "concentration_percent": 2.0}
                    ],
                    "nitrogen_source": ["yeast_extract", "peptone"],
                }
            }
        }
        result = mgr._flatten_condition_definition(definition)
        assert result["growth_media"] == "YPD"
        assert "carbon_source:D-glucose@2.0%" in result["components"]
        assert "nitrogen_source:yeast_extract" in result["components"]
        assert "nitrogen_source:peptone" in result["components"]


class TestComponentSeparators:
    """Tests for separator conventions."""

    def test_separator_constants(self):
        """Test that separator constants are defined."""
        from tfbpapi.datainfo.metadata_manager import COMPONENT_SEPARATORS

        assert COMPONENT_SEPARATORS["type_value"] == ":"
        assert COMPONENT_SEPARATORS["value_conc"] == "@"
        assert COMPONENT_SEPARATORS["components"] == ";"
        assert COMPONENT_SEPARATORS["types"] == "|"


class TestThreeLevelConditionHierarchy:
    """Tests for three-level experimental condition hierarchy support."""

    def test_flatten_experimental_conditions_empty(self):
        """Test flattening empty ExperimentalConditions."""
        mgr = MetadataManager()

        # Create a simple mock object with no conditions
        class MockExpConditions:
            environmental_conditions = None
            strain_background = None

        result = mgr._flatten_experimental_conditions(MockExpConditions())
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_flatten_experimental_conditions_with_temperature(self):
        """Test flattening conditions with temperature."""
        mgr = MetadataManager()

        class MockEnv:
            temperature_celsius = 30
            cultivation_method = None
            media = None
            growth_phase = None
            chemical_treatments = None
            drug_treatments = None
            heat_treatment = None
            induction = None

        class MockExpConditions:
            environmental_conditions = MockEnv()
            strain_background = None

        result = mgr._flatten_experimental_conditions(MockExpConditions())
        assert result["temperature_celsius"] == 30

    def test_flatten_experimental_conditions_with_cultivation_method(self):
        """Test flattening conditions with cultivation method."""
        mgr = MetadataManager()

        class MockEnv:
            temperature_celsius = None
            cultivation_method = "chemostat"
            media = None
            growth_phase = None
            chemical_treatments = None
            drug_treatments = None
            heat_treatment = None
            induction = None

        class MockExpConditions:
            environmental_conditions = MockEnv()
            strain_background = None

        result = mgr._flatten_experimental_conditions(MockExpConditions())
        assert result["cultivation_method"] == "chemostat"

    def test_flatten_experimental_conditions_with_media(self):
        """Test flattening conditions with media information."""
        mgr = MetadataManager()

        class MockCompound:
            compound = "D-glucose"
            concentration_percent = 1.0

            def model_dump(self):
                return {
                    "name": self.compound,
                    "concentration_percent": self.concentration_percent,
                }

        class MockMedia:
            name = "minimal"
            carbon_source = [MockCompound()]
            nitrogen_source = None
            phosphate_source = None
            additives = None

        class MockEnv:
            temperature_celsius = None
            cultivation_method = None
            media = MockMedia()
            growth_phase = None
            chemical_treatments = None
            drug_treatments = None
            heat_treatment = None
            induction = None

        class MockExpConditions:
            environmental_conditions = MockEnv()
            strain_background = None

        result = mgr._flatten_experimental_conditions(MockExpConditions())
        assert result["growth_media"] == "minimal"
        assert "carbon_source:D-glucose@1.0%" in result["components"]

    def test_flatten_experimental_conditions_with_strain_background(self):
        """Test flattening conditions with strain background."""
        mgr = MetadataManager()

        class MockExpConditions:
            environmental_conditions = None
            strain_background = "BY4741"

        result = mgr._flatten_experimental_conditions(MockExpConditions())
        assert result["strain_background"] == "BY4741"

    def test_datacard_extract_schema_includes_top_level_conditions(self):
        """Test that extract_metadata_schema includes top_level_conditions."""
        # This test needs to use real datacards with experimental conditions
        # For now, we verify the keys exist in the schema
        from unittest.mock import MagicMock, Mock

        # Create a mock DataCard with experimental conditions
        mock_card = Mock()
        mock_card.repo_id = "test/repo"

        # Mock dataset card with experimental conditions
        mock_dataset_card = Mock()
        mock_dataset_card.experimental_conditions = Mock()  # Non-None
        mock_card.dataset_card = mock_dataset_card

        # Mock config with no config-level conditions
        mock_config = Mock()
        mock_config.experimental_conditions = None
        mock_config.dataset_info = Mock()
        mock_config.dataset_info.features = []

        mock_card.get_config = Mock(return_value=mock_config)

        # Call the method via the actual DataCard class
        from tfbpapi.datainfo.datacard import DataCard

        schema = DataCard.extract_metadata_schema(mock_card, "test_config")

        # Verify top_level_conditions is in schema
        assert "top_level_conditions" in schema
        assert "config_level_conditions" in schema

    def test_datacard_extract_schema_includes_config_level_conditions(self):
        """Test that extract_metadata_schema includes config_level_conditions."""
        from unittest.mock import Mock

        # Create a mock DataCard
        mock_card = Mock()
        mock_card.repo_id = "test/repo"

        # Mock dataset card with NO repo-level conditions
        mock_dataset_card = Mock()
        mock_dataset_card.experimental_conditions = None
        mock_card.dataset_card = mock_dataset_card

        # Mock config WITH config-level conditions
        mock_config = Mock()
        mock_config.experimental_conditions = Mock()  # Non-None
        mock_config.dataset_info = Mock()
        mock_config.dataset_info.features = []

        mock_card.get_config = Mock(return_value=mock_config)

        # Call the method
        from tfbpapi.datainfo.datacard import DataCard

        schema = DataCard.extract_metadata_schema(mock_card, "test_config")

        # Verify config_level_conditions is populated
        assert schema["config_level_conditions"] is not None


# Integration test placeholder
class TestMetadataManagerIntegration:
    """Integration tests for MetadataManager (require real HF datasets)."""

    @pytest.mark.skip(reason="Requires real HuggingFace dataset access")
    def test_register_real_dataset(self):
        """Test registering a real HuggingFace dataset."""
        # This would test with a real dataset like:
        # mgr = MetadataManager()
        # mgr.register("BrentLab/some_real_dataset")
        # assert len(mgr.get_active_configs()) > 0
        pass

    @pytest.mark.skip(reason="Requires real HuggingFace dataset access")
    def test_query_across_datasets(self):
        """Test querying across multiple datasets."""
        # This would test cross-dataset queries
        pass
