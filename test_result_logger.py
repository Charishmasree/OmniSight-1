"""
Unit tests for result_logger.py
Verifies saving reports to JSON, loading reports, generating summaries,
and handling output directories.
"""

import json
import tempfile
import unittest
from pathlib import Path

from prompts import get_mock_analysis_data
from result_logger import (
    save_analysis_report,
    load_analysis_report,
    format_summary_report,
    generate_default_filename,
)


class TestResultLogger(unittest.TestCase):
    """Test cases for saving, loading, and formatting analysis results."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.sample_data = get_mock_analysis_data(screenshot_path="home_screen.png")
        self.sample_data["metadata"] = {
            "timestamp": "2026-10-16T07:00:00Z",
            "screenshot_path": "screenshots/home_screen.png",
            "model": "gemini-2.5-flash",
            "status": "success",
            "duration_seconds": 1.25,
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_load_analysis_report(self):
        """Should save analysis dictionary to JSON and reload exact data."""
        saved_file = save_analysis_report(
            data=self.sample_data,
            output_dir=self.output_dir,
            filename="test_report.json",
        )

        self.assertTrue(saved_file.exists())
        self.assertEqual(saved_file.name, "test_report.json")

        loaded_data = load_analysis_report(saved_file)
        self.assertEqual(loaded_data["visual_health_score"], self.sample_data["visual_health_score"])
        self.assertEqual(loaded_data["page_summary"], self.sample_data["page_summary"])
        self.assertEqual(len(loaded_data["visual_issues"]), len(self.sample_data["visual_issues"]))

    def test_save_with_explicit_output_path(self):
        """Should respect explicit full output_path."""
        explicit_path = self.output_dir / "nested" / "deep" / "custom.json"
        saved_file = save_analysis_report(
            data=self.sample_data,
            output_path=explicit_path,
        )
        self.assertTrue(saved_file.exists())
        self.assertEqual(saved_file, explicit_path)

    def test_save_with_auto_generated_filename(self):
        """Should auto-generate filename from screenshot_path in metadata."""
        saved_file = save_analysis_report(
            data=self.sample_data,
            output_dir=self.output_dir,
        )
        self.assertTrue(saved_file.exists())
        self.assertTrue(saved_file.name.startswith("home_screen_analysis_"))
        self.assertTrue(saved_file.name.endswith(".json"))

    def test_load_nonexistent_file_raises_error(self):
        """Should raise FileNotFoundError when file does not exist."""
        nonexistent = self.output_dir / "does_not_exist.json"
        with self.assertRaises(FileNotFoundError):
            load_analysis_report(nonexistent)

    def test_format_summary_report(self):
        """Should build readable summary text with element counts and issues."""
        summary = format_summary_report(self.sample_data)
        self.assertIn("OMNISIGHT UI/UX VISION ANALYSIS REPORT", summary)
        self.assertIn("Health Score:    78/100", summary)
        self.assertIn("Buttons:", summary)
        self.assertIn("Input Fields:", summary)
        self.assertIn("Visual Issues Detected:", summary)
        self.assertIn("Hero subtitle overlaps primary CTA button", summary)
        self.assertIn("Low contrast on footer newsletter disclaimer", summary)


if __name__ == "__main__":
    unittest.main()
