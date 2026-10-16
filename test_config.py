"""
Unit tests for config.py
Verifies configuration loading, defaults, environment variable overrides,
and validation logic.
"""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from config import (
    VisionConfig,
    get_config,
    APIKeyMissingError,
    OmniSightVisionError,
    ImageNotFoundError,
    InvalidImageFormatError,
    ModelInferenceError,
)


class TestVisionConfig(unittest.TestCase):
    """Test cases for VisionConfig."""

    def test_default_values(self):
        """Verifies default configuration values."""
        cfg = VisionConfig(gemini_api_key="test-key", mock_mode=False)
        self.assertEqual(cfg.model_name, "gemini-2.5-flash")
        self.assertEqual(cfg.output_dir, Path("analysis_reports"))
        self.assertIn(".png", cfg.supported_formats)
        self.assertIn(".jpg", cfg.supported_formats)
        self.assertIn(".webp", cfg.supported_formats)
        self.assertEqual(cfg.temperature, 0.2)
        self.assertEqual(cfg.max_image_size_mb, 15.0)

    def test_validation_raises_when_no_api_key_and_not_mock(self):
        """Should raise APIKeyMissingError if gemini_api_key is empty and mock_mode is False."""
        cfg = VisionConfig(gemini_api_key="", mock_mode=False)
        with self.assertRaises(APIKeyMissingError):
            cfg.validate()

    def test_validation_passes_in_mock_mode_without_api_key(self):
        """Should succeed validation in mock mode even without an API key."""
        cfg = VisionConfig(gemini_api_key="", mock_mode=True)
        # Should not raise any exception
        cfg.validate()
        self.assertTrue(cfg.mock_mode)

    def test_validation_passes_with_api_key(self):
        """Should succeed validation when API key is provided."""
        cfg = VisionConfig(gemini_api_key="AIzaSyFakeKeyForTesting12345", mock_mode=False)
        cfg.validate()
        self.assertEqual(cfg.gemini_api_key, "AIzaSyFakeKeyForTesting12345")

    def test_get_config_with_overrides(self):
        """Factory function should respect overrides."""
        custom_dir = Path("custom_reports")
        cfg = get_config(
            gemini_api_key="override-key",
            model_name="gemini-2.0-flash",
            output_dir=custom_dir,
            mock_mode=True,
        )
        self.assertEqual(cfg.gemini_api_key, "override-key")
        self.assertEqual(cfg.model_name, "gemini-2.0-flash")
        self.assertEqual(cfg.output_dir, custom_dir)
        self.assertTrue(cfg.mock_mode)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "env-secret-key", "VLM_MODEL_NAME": "custom-model", "VLM_MOCK_MODE": "true"})
    def test_environment_variable_loading(self):
        """Should read configuration values from os.environ."""
        cfg = VisionConfig()
        self.assertEqual(cfg.gemini_api_key, "env-secret-key")
        self.assertEqual(cfg.model_name, "custom-model")
        self.assertTrue(cfg.mock_mode)

    def test_exception_hierarchy(self):
        """Ensure all custom exceptions inherit from OmniSightVisionError."""
        self.assertTrue(issubclass(APIKeyMissingError, OmniSightVisionError))
        self.assertTrue(issubclass(ImageNotFoundError, OmniSightVisionError))
        self.assertTrue(issubclass(InvalidImageFormatError, OmniSightVisionError))
        self.assertTrue(issubclass(ModelInferenceError, OmniSightVisionError))


if __name__ == "__main__":
    unittest.main()
