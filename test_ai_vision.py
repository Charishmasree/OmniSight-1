"""
Unit tests for ai_vision.py
Verifies image validation, JSON parsing, mock inference, error handling,
and mocked Google Gemini API calls without requiring real API keys or internet access.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from ai_vision import VisionAnalyzer, analyze_and_save
from config import (
    VisionConfig,
    APIKeyMissingError,
    ImageNotFoundError,
    InvalidImageFormatError,
    ImageCorruptedError,
    ImageSizeExceededError,
    ModelInferenceError,
    ResponseParseError,
)


class TestVisionAnalyzer(unittest.TestCase):
    """Test suite for VisionAnalyzer class and helper functions."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

        # Create a valid test image
        self.valid_image_path = self.dir_path / "valid_test.png"
        img = Image.new("RGB", (200, 100), color=(100, 150, 200))
        img.save(self.valid_image_path, format="PNG")

    def tearDown(self):
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # Image Validation Tests
    # -------------------------------------------------------------------------

    def test_validate_image_success(self):
        """Should successfully validate an existing, well-formed PNG image."""
        analyzer = VisionAnalyzer(mock_mode=True)
        validated = analyzer.validate_image(self.valid_image_path)
        self.assertEqual(validated, self.valid_image_path.resolve())

    def test_validate_image_nonexistent_raises_not_found(self):
        """Should raise ImageNotFoundError when screenshot does not exist."""
        analyzer = VisionAnalyzer(mock_mode=True)
        fake_path = self.dir_path / "ghost_screenshot.png"
        with self.assertRaises(ImageNotFoundError):
            analyzer.validate_image(fake_path)

    def test_validate_image_invalid_extension_raises_format_error(self):
        """Should raise InvalidImageFormatError for unsupported file types."""
        analyzer = VisionAnalyzer(mock_mode=True)
        text_file = self.dir_path / "document.txt"
        text_file.write_text("not an image", encoding="utf-8")
        with self.assertRaises(InvalidImageFormatError):
            analyzer.validate_image(text_file)

    def test_validate_image_corrupted_raises_corrupted_error(self):
        """Should raise ImageCorruptedError when file has .png extension but invalid image bytes."""
        analyzer = VisionAnalyzer(mock_mode=True)
        corrupted_png = self.dir_path / "corrupted.png"
        corrupted_png.write_bytes(b"This is fake data and not a valid PNG header.")
        with self.assertRaises(ImageCorruptedError):
            analyzer.validate_image(corrupted_png)

    def test_validate_image_size_exceeded_raises_size_error(self):
        """Should raise ImageSizeExceededError if image file exceeds max_image_size_mb."""
        cfg = VisionConfig(mock_mode=True, max_image_size_mb=0.0001)  # ~100 bytes limit
        analyzer = VisionAnalyzer(config=cfg)
        with self.assertRaises(ImageSizeExceededError):
            analyzer.validate_image(self.valid_image_path)

    def test_get_image_info(self):
        """Should accurately read dimensions and format."""
        analyzer = VisionAnalyzer(mock_mode=True)
        info = analyzer.get_image_info(self.valid_image_path)
        self.assertEqual(info["width"], 200)
        self.assertEqual(info["height"], 100)
        self.assertEqual(info["format"], "PNG")
        self.assertGreater(info["file_size_bytes"], 0)

    # -------------------------------------------------------------------------
    # JSON Parsing & Cleaning Tests
    # -------------------------------------------------------------------------

    def test_clean_and_parse_raw_json(self):
        """Should parse raw valid JSON string."""
        analyzer = VisionAnalyzer(mock_mode=True)
        raw_json = '{"page_summary": "Test Page", "visual_health_score": 90, "ui_elements": {}, "visual_issues": []}'
        parsed = analyzer._clean_and_parse_json(raw_json)
        self.assertEqual(parsed["visual_health_score"], 90)
        self.assertEqual(parsed["page_summary"], "Test Page")

    def test_clean_and_parse_markdown_fenced_json(self):
        """Should strip markdown code fences (```json ... ```) and parse cleanly."""
        analyzer = VisionAnalyzer(mock_mode=True)
        fenced_json = """```json
{
  "page_summary": "Fenced JSON Page",
  "visual_health_score": 85,
  "ui_elements": {"buttons": [{"label": "Submit"}]},
  "visual_issues": []
}
```"""
        parsed = analyzer._clean_and_parse_json(fenced_json)
        self.assertEqual(parsed["visual_health_score"], 85)
        self.assertEqual(parsed["ui_elements"]["buttons"][0]["label"], "Submit")

    def test_clean_and_parse_invalid_json_raises_error(self):
        """Should raise ResponseParseError when model response is malformed."""
        analyzer = VisionAnalyzer(mock_mode=True)
        bad_json = "Sorry, I cannot analyze this image properly as JSON."
        with self.assertRaises(ResponseParseError):
            analyzer._clean_and_parse_json(bad_json)

    def test_clean_and_parse_non_dict_json_raises_error(self):
        """Should raise ResponseParseError if JSON is an array rather than an object."""
        analyzer = VisionAnalyzer(mock_mode=True)
        array_json = '[{"issue": "test"}]'
        with self.assertRaises(ResponseParseError):
            analyzer._clean_and_parse_json(array_json)

    # -------------------------------------------------------------------------
    # Initialization & Auth Tests
    # -------------------------------------------------------------------------

    def test_init_without_api_key_raises_in_live_mode(self):
        """Should raise APIKeyMissingError if mock_mode=False and no API key is provided."""
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(APIKeyMissingError):
                VisionAnalyzer(api_key="", mock_mode=False)

    # -------------------------------------------------------------------------
    # Mock Mode Inference Tests (Zero API Calls)
    # -------------------------------------------------------------------------

    def test_analyze_screenshot_in_mock_mode(self):
        """Mock mode should return complete structured data with metadata and zero API calls."""
        analyzer = VisionAnalyzer(mock_mode=True)
        result = analyzer.analyze_screenshot(self.valid_image_path)

        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["status"], "success")
        self.assertTrue(result["metadata"]["mock_mode"])
        self.assertEqual(result["metadata"]["image_dimensions"]["width"], 200)
        self.assertEqual(result["metadata"]["image_dimensions"]["height"], 100)

        self.assertIn("ui_elements", result)
        self.assertIn("visual_issues", result)
        self.assertIn("page_summary", result)
        self.assertIsInstance(result["visual_health_score"], int)

    # -------------------------------------------------------------------------
    # Mocked External API Calls (Unit Testing VLM Integration)
    # -------------------------------------------------------------------------

    @patch("google.genai.Client")
    def test_mocked_gemini_api_success(self, mock_client_class):
        """Simulate a successful external Gemini VLM response using a mocked client."""
        # Setup mock client
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance

        # Simulated response from Google Gemini
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "page_summary": "Mocked Gemini Response for Automated Test",
            "visual_health_score": 92,
            "ui_elements": {
                "buttons": [{"label": "Login", "type": "primary", "location": "navbar", "state": "normal"}],
                "input_fields": [],
                "text_elements": [],
                "navigation_elements": [],
                "media_elements": []
            },
            "visual_issues": [
                {
                    "issue_id": "MOCK-001",
                    "category": "overlapping_elements",
                    "severity": "low",
                    "title": "Minor badge overlap",
                    "location": "Header",
                    "description": "Badge touches nav border slightly.",
                    "recommendation": "Add 2px padding."
                }
            ],
            "recommendations": ["Refine header padding."]
        })
        mock_client_instance.models.generate_content.return_value = mock_response

        # Initialize analyzer with dummy key and mocked client
        analyzer = VisionAnalyzer(api_key="mock-api-key-12345", mock_mode=False)
        result = analyzer.analyze_screenshot(self.valid_image_path)

        # Verify client was called with expected model and contents
        mock_client_instance.models.generate_content.assert_called_once()
        self.assertEqual(result["page_summary"], "Mocked Gemini Response for Automated Test")
        self.assertEqual(result["visual_health_score"], 92)
        self.assertEqual(result["metadata"]["status"], "success")
        self.assertEqual(result["metadata"]["model"], analyzer.config.model_name)
        self.assertEqual(len(result["visual_issues"]), 1)
        self.assertEqual(result["visual_issues"][0]["category"], "overlapping_elements")

    @patch("google.genai.Client")
    def test_mocked_gemini_api_network_failure(self, mock_client_class):
        """Simulate an external API error (e.g. timeout or 503) wrapping into ModelInferenceError."""
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.models.generate_content.side_effect = Exception("503 Service Unavailable")

        analyzer = VisionAnalyzer(api_key="mock-api-key-12345", mock_mode=False)
        with self.assertRaises(ModelInferenceError) as ctx:
            analyzer.analyze_screenshot(self.valid_image_path)

        self.assertIn("503 Service Unavailable", str(ctx.exception))

    @patch("google.genai.Client")
    def test_mocked_gemini_empty_response(self, mock_client_class):
        """Simulate empty response from model."""
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_response = MagicMock()
        mock_response.text = None
        mock_client_instance.models.generate_content.return_value = mock_response

        analyzer = VisionAnalyzer(api_key="mock-api-key-12345", mock_mode=False)
        with self.assertRaises(ModelInferenceError):
            analyzer.analyze_screenshot(self.valid_image_path)

    # -------------------------------------------------------------------------
    # Convenience Function Integration
    # -------------------------------------------------------------------------

    def test_analyze_and_save_convenience_function(self):
        """Should analyze screenshot and save report in one unified call."""
        out_dir = self.dir_path / "reports"
        result, saved_file = analyze_and_save(
            image_path=self.valid_image_path,
            output_dir=out_dir,
            filename="quick_test.json",
            mock_mode=True,
        )

        self.assertTrue(saved_file.exists())
        self.assertEqual(saved_file.name, "quick_test.json")
        self.assertIn("ui_elements", result)


if __name__ == "__main__":
    unittest.main()




import unittest
from pathlib import Path


class TestVisionFiles(unittest.TestCase):

    def test_sample_image_exists(self):
        image_path = Path("sample_ui.png")
        self.assertTrue(
            image_path.exists(),
            "Sample UI image should exist"
        )

    def test_sample_image_is_not_empty(self):
        image_path = Path("sample_ui.png")
        if image_path.exists():
            self.assertGreater(
                image_path.stat().st_size, 0,
                "Sample image should not be empty"
            )

    def test_vision_module_exists(self):
        vision_path = Path("ai_vision.py")
        self.assertTrue(
            vision_path.exists(),
            "AI vision module should exist"
        )


if __name__ == "__main__":
    unittest.main()

