"""
Unit tests for prompts.py
Verifies prompt construction, schema inclusion, focus areas, and mock data integrity.
"""

import unittest
from prompts import (
    SYSTEM_PROMPT,
    JSON_SCHEMA_INSTRUCTIONS,
    build_analysis_prompt,
    get_mock_analysis_data,
)


class TestPrompts(unittest.TestCase):
    """Test cases for prompt generation and schema formatting."""

    def test_default_prompt_contains_system_instructions(self):
        """Prompt should contain system role and key tasks."""
        prompt = build_analysis_prompt()
        self.assertIn("Senior Automated QA & UI/UX Accessibility Vision Inspector", prompt)
        self.assertIn("Overlapping elements", prompt)
        self.assertIn("Unreadable text", prompt)
        self.assertIn("Missing visible elements", prompt)
        self.assertIn("Layout issues", prompt)
        self.assertIn("JSON", prompt)

    def test_prompt_includes_focus_areas(self):
        """Prompt should format and inject specific focus areas."""
        focus = ["buttons", "overlapping_elements", "mobile_nav"]
        prompt = build_analysis_prompt(focus_areas=focus)
        self.assertIn("High-Priority Focus Areas", prompt)
        for area in focus:
            self.assertIn(f"- {area}", prompt)

    def test_prompt_includes_custom_notes(self):
        """Prompt should inject custom test notes."""
        notes = "Check the checkout modal on an iPad Mini resolution (768x1024)."
        prompt = build_analysis_prompt(custom_notes=notes)
        self.assertIn("Additional Test Context:", prompt)
        self.assertIn(notes, prompt)

    def test_mock_data_structure(self):
        """Mock data should adhere to the expected schema with all required keys."""
        mock_data = get_mock_analysis_data(screenshot_path="tests/fixtures/sample.png")

        # Top-level keys
        self.assertIn("page_summary", mock_data)
        self.assertIn("visual_health_score", mock_data)
        self.assertIn("ui_elements", mock_data)
        self.assertIn("visual_issues", mock_data)
        self.assertIn("recommendations", mock_data)

        # Health score range
        self.assertIsInstance(mock_data["visual_health_score"], int)
        self.assertTrue(0 <= mock_data["visual_health_score"] <= 100)

        # UI elements categories
        ui = mock_data["ui_elements"]
        self.assertIn("buttons", ui)
        self.assertIn("input_fields", ui)
        self.assertIn("text_elements", ui)
        self.assertIn("navigation_elements", ui)
        self.assertIn("media_elements", ui)
        self.assertGreater(len(ui["buttons"]), 0)

        # Visual issues
        issues = mock_data["visual_issues"]
        self.assertGreater(len(issues), 0)
        first_issue = issues[0]
        self.assertIn("issue_id", first_issue)
        self.assertIn("category", first_issue)
        self.assertIn("severity", first_issue)
        self.assertIn("title", first_issue)
        self.assertIn("location", first_issue)
        self.assertIn("description", first_issue)
        self.assertIn("recommendation", first_issue)


if __name__ == "__main__":
    unittest.main()
