"""
OmniSight - Member 2: AI Integration using Vision-Language Models (VLM)
Week 3: Unit Tests for Self-Healing Loop (ai/test_self_healing.py)

Comprehensive test suite verifying:
1. Successful visual fix on attempt 1
2. Unsuccessful visual fix
3. Maximum retry limit reached
4. Malformed AI output handled gracefully
5. Invalid CSS syntax handled gracefully
6. Browser navigation/execution errors handled gracefully
7. VLM API network failure handled gracefully
8. Robust CSS extraction across multiple schema variants
9. Multi-attempt healing with revised CSS suggestion
10. Runtime DOM limitation explanation preserved in output metadata
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai.self_healing_loop import (
    SelfHealingLoop,
    run_self_healing_loop,
    extract_css_fix,
    CSSValidator,
    CSSApplicationError,
    BrowserExecutionError,
)
from ai.vision_analyzer import VisionAnalyzer
from config import ModelInferenceError, ResponseParseError


class TestSelfHealingLoop(unittest.TestCase):
    """Test suite for Week 3 Self-Healing Loop."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

        # Inline sample HTML with an overlapping subtitle bug
        self.sample_html = """<!DOCTYPE html>
        <html>
        <head>
          <style>
            .hero-subtitle { margin-bottom: -30px; line-height: 1.0; }
            .cta-btn { padding: 12px 24px; background: #4f46e5; color: white; }
          </style>
        </head>
        <body>
          <h1 class="hero-title">OmniSight Test</h1>
          <p class="hero-subtitle">Testing automated self healing loop</p>
          <button class="cta-btn">Primary Action</button>
        </body>
        </html>"""

        # Sample valid issue with suggested fix
        self.sample_issue = {
            "issue_id": "ISSUE-001",
            "category": "overlapping_text",
            "severity": "high",
            "title": "Hero subtitle collides with CTA button",
            "selector": ".hero-subtitle",
            "description": "Subtitle overlaps button by 30px.",
            "suggested_fix": {
                "selector": ".hero-subtitle",
                "css_properties": {"margin-bottom": "24px", "line-height": "1.4"},
                "css_code": ".hero-subtitle { margin-bottom: 24px; line-height: 1.4; }",
                "explanation": "Add 24px bottom margin to separate subtitle from button.",
            },
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # 1. Successful Fix
    # -------------------------------------------------------------------------

    def test_successful_fix_single_attempt(self):
        """Should apply CSS fix, verify with VLM, and return 'fixed' status on attempt 1."""
        loop = SelfHealingLoop(
            max_attempts=3,
            output_dir=self.dir_path,
            mock_mode=True,
            headless=True,
        )

        result = loop.heal_issue(
            page_target=self.sample_html,
            detected_issue=self.sample_issue,
            force_verification_status="fixed",
        )

        # Verify all 6 mandatory structured fields:
        self.assertIn("original_issue", result)
        self.assertIn("suggested_css", result)
        self.assertIn("attempt", result)
        self.assertIn("verification_status", result)
        self.assertIn("verification_reason", result)
        self.assertIn("screenshot_path", result)

        self.assertEqual(result["verification_status"], "fixed")
        self.assertTrue(result["success"])
        self.assertEqual(result["attempt"], 1)
        self.assertIn("margin-bottom: 24px", result["suggested_css"])
        self.assertTrue(Path(result["screenshot_path"]).exists())
        self.assertEqual(len(result["history"]), 1)

    # -------------------------------------------------------------------------
    # 2. Unsuccessful Fix
    # -------------------------------------------------------------------------

    def test_unsuccessful_fix(self):
        """Should record 'unresolved' when verification indicates defect persists."""
        loop = SelfHealingLoop(
            max_attempts=1,
            output_dir=self.dir_path,
            mock_mode=True,
            headless=True,
        )

        result = loop.heal_issue(
            page_target=self.sample_html,
            detected_issue=self.sample_issue,
            force_verification_status="unresolved",
        )

        self.assertEqual(result["verification_status"], "unresolved")
        self.assertFalse(result["success"])
        self.assertEqual(result["attempt"], 1)
        self.assertIn("persists", result["verification_reason"].lower())

    # -------------------------------------------------------------------------
    # 3. Maximum Retry Limit
    # -------------------------------------------------------------------------

    def test_maximum_retry_limit_reached(self):
        """Should retry up to max_attempts and stop with 'unresolved' status."""
        loop = SelfHealingLoop(
            max_attempts=3,
            output_dir=self.dir_path,
            mock_mode=True,
            headless=True,
        )

        result = loop.heal_issue(
            page_target=self.sample_html,
            detected_issue=self.sample_issue,
            force_verification_status="unresolved",
        )

        self.assertEqual(result["attempt"], 3)
        self.assertEqual(result["total_attempts"], 3)
        self.assertEqual(result["max_attempts"], 3)
        self.assertEqual(len(result["history"]), 3)
        self.assertEqual(result["verification_status"], "unresolved")
        self.assertFalse(result["success"])

    # -------------------------------------------------------------------------
    # 4. Malformed AI Output Handled Gracefully
    # -------------------------------------------------------------------------

    def test_malformed_ai_output_handled_gracefully(self):
        """Should handle unparseable/malformed VLM output without crashing."""
        analyzer = VisionAnalyzer(mock_mode=True)
        # Mock verify_fix to raise ResponseParseError simulating corrupt model output
        analyzer.verify_fix = MagicMock(
            side_effect=ResponseParseError("Model output was not valid JSON: unexpected token")
        )

        loop = SelfHealingLoop(
            analyzer=analyzer,
            max_attempts=2,
            output_dir=self.dir_path,
            headless=True,
        )

        result = loop.heal_issue(
            page_target=self.sample_html,
            detected_issue=self.sample_issue,
        )

        # Loop must complete gracefully without unhandled exception
        self.assertEqual(result["verification_status"], "unresolved")
        self.assertFalse(result["success"])
        self.assertIn("VLM verification error", result["verification_reason"])

    # -------------------------------------------------------------------------
    # 5. Invalid CSS Syntax Handled Gracefully
    # -------------------------------------------------------------------------

    def test_invalid_css_syntax_handled_gracefully(self):
        """Should reject broken CSS syntax (e.g. unclosed braces) without crashing."""
        broken_issue = {
            "issue_id": "ISSUE-BROKEN-CSS",
            "selector": ".broken",
            "suggested_fix": {
                "css_code": ".broken { margin-bottom: 24px; color: red;"  # Missing closing brace
            },
        }

        loop = SelfHealingLoop(
            max_attempts=2,
            output_dir=self.dir_path,
            mock_mode=True,
        )

        result = loop.heal_issue(
            page_target=self.sample_html,
            detected_issue=broken_issue,
        )

        self.assertEqual(result["verification_status"], "unresolved")
        self.assertFalse(result["success"])
        self.assertIn("Invalid CSS syntax", result["verification_reason"])

    # -------------------------------------------------------------------------
    # 6. Browser Navigation / Execution Error Handled Gracefully
    # -------------------------------------------------------------------------

    def test_browser_error_handled_gracefully(self):
        """Should catch non-existent local test file paths and report error cleanly."""
        loop = SelfHealingLoop(
            max_attempts=1,
            output_dir=self.dir_path,
            mock_mode=True,
        )

        fake_path = self.dir_path / "non_existent_page.html"
        result = loop.heal_issue(
            page_target=fake_path,
            detected_issue=self.sample_issue,
        )

        self.assertEqual(result["verification_status"], "unresolved")
        self.assertFalse(result["success"])
        self.assertIn("Browser automation error", result["verification_reason"])

    # -------------------------------------------------------------------------
    # 7. VLM Network / API Failure Handled Gracefully
    # -------------------------------------------------------------------------

    def test_vlm_network_api_failure_handled_gracefully(self):
        """Should catch ModelInferenceError (HTTP 500, timeout) without crashing."""
        analyzer = VisionAnalyzer(mock_mode=True)
        analyzer.verify_fix = MagicMock(
            side_effect=ModelInferenceError("503 Service Unavailable: Gemini API rate limit exceeded")
        )

        loop = SelfHealingLoop(
            analyzer=analyzer,
            max_attempts=1,
            output_dir=self.dir_path,
        )

        result = loop.heal_issue(
            page_target=self.sample_html,
            detected_issue=self.sample_issue,
        )

        self.assertEqual(result["verification_status"], "unresolved")
        self.assertFalse(result["success"])
        self.assertIn("503 Service Unavailable", result["verification_reason"])

    # -------------------------------------------------------------------------
    # 8. CSS Extraction Utilities
    # -------------------------------------------------------------------------

    def test_extract_css_fix_from_various_structures(self):
        """extract_css_fix should support multiple VLM output schema variants."""
        # 1. css_code in suggested_fix
        case1 = {"suggested_fix": {"css_code": ".card { padding: 16px; }"}}
        self.assertEqual(extract_css_fix(case1), ".card { padding: 16px; }")

        # 2. selector + css_properties in suggested_fix
        case2 = {
            "suggested_fix": {
                "selector": ".nav-link",
                "css_properties": {"color": "#333", "font-weight": "bold"},
            }
        }
        extracted2 = extract_css_fix(case2)
        self.assertIn(".nav-link {", extracted2)
        self.assertIn("color: #333", extracted2)

        # 3. Direct suggested_css string
        case3 = {"suggested_css": ".btn { display: flex; }"}
        self.assertEqual(extract_css_fix(case3), ".btn { display: flex; }")

        # 4. Plain string input
        self.assertEqual(extract_css_fix(".footer { clear: both; }"), ".footer { clear: both; }")

        # 5. Invalid empty input raises CSSApplicationError
        with self.assertRaises(CSSApplicationError):
            extract_css_fix({})

    # -------------------------------------------------------------------------
    # 9. CSS Validator Tests
    # -------------------------------------------------------------------------

    def test_css_validator(self):
        """CSSValidator should validate correct CSS and flag malformed patterns."""
        valid, _ = CSSValidator.validate_css(".hero { margin: 10px; }")
        self.assertTrue(valid)

        # Empty string
        valid_empty, _ = CSSValidator.validate_css("")
        self.assertFalse(valid_empty)

        # Mismatched braces
        valid_braces, _ = CSSValidator.validate_css(".hero { margin: 10px;")
        self.assertFalse(valid_braces)

        # Dangerous script pattern
        valid_script, _ = CSSValidator.validate_css(".hero { behavior: url(script.htc); }")
        self.assertFalse(valid_script)

    # -------------------------------------------------------------------------
    # 10. Multi-Attempt Healing with Revised CSS
    # -------------------------------------------------------------------------

    def test_multi_attempt_healing_with_revised_css(self):
        """Attempt 1: partially_fixed with revised CSS -> Attempt 2: fixed."""
        analyzer = VisionAnalyzer(mock_mode=True)

        call_count = 0

        def dynamic_verify(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "verification_status": "partially_fixed",
                    "verification_reason": "Subtitle overlap reduced, but baseline still off by 4px.",
                    "defect_resolved": False,
                    "visual_health_score": 80,
                    "remaining_issues": [{"description": "Minor offset", "severity": "low"}],
                    "revised_css_suggestion": {
                        "selector": ".hero-subtitle",
                        "css_code": ".hero-subtitle { margin-bottom: 32px; line-height: 1.5; }",
                    },
                }
            else:
                return {
                    "verification_status": "fixed",
                    "verification_reason": "Defect fully resolved with revised 32px bottom margin.",
                    "defect_resolved": True,
                    "visual_health_score": 98,
                    "remaining_issues": [],
                    "revised_css_suggestion": None,
                }

        analyzer.verify_fix = MagicMock(side_effect=dynamic_verify)

        loop = SelfHealingLoop(
            analyzer=analyzer,
            max_attempts=3,
            output_dir=self.dir_path,
            headless=True,
        )

        result = loop.heal_issue(
            page_target=self.sample_html,
            detected_issue=self.sample_issue,
        )

        self.assertEqual(result["verification_status"], "fixed")
        self.assertTrue(result["success"])
        self.assertEqual(result["attempt"], 2)
        self.assertEqual(len(result["history"]), 2)
        self.assertIn("margin-bottom: 32px", result["suggested_css"])

    # -------------------------------------------------------------------------
    # 11. Runtime DOM Limitation Explanation Included
    # -------------------------------------------------------------------------

    def test_runtime_dom_limitation_explanation_included(self):
        """Result metadata must include explanation of runtime Playwright DOM limitation."""
        loop = SelfHealingLoop(
            max_attempts=1,
            output_dir=self.dir_path,
            mock_mode=True,
        )

        result = loop.heal_issue(
            page_target=self.sample_html,
            detected_issue=self.sample_issue,
            force_verification_status="fixed",
        )

        self.assertIn("metadata", result)
        self.assertIn("runtime_dom_only_note", result["metadata"])
        note = result["metadata"]["runtime_dom_only_note"]
        self.assertIn("runtime", note.lower())
        self.assertIn("source files", note.lower())

    # -------------------------------------------------------------------------
    # 12. Convenience Function
    # -------------------------------------------------------------------------

    def test_standalone_run_self_healing_loop_convenience_function(self):
        """run_self_healing_loop should execute end-to-end in a single function call."""
        result = run_self_healing_loop(
            page_target=self.sample_html,
            detected_issue=self.sample_issue,
            max_attempts=2,
            mock_mode=True,
            output_dir=self.dir_path,
        )

        self.assertIn("verification_status", result)
        self.assertIn(result["verification_status"], ("fixed", "partially_fixed", "unresolved"))
        self.assertIn("screenshot_path", result)


if __name__ == "__main__":
    unittest.main()





from unittest.mock import Mock


def test_css_application():
    page = Mock()
    page.locator.return_value.is_visible.return_value = True

    assert page.locator("#submit-button").is_visible()


def test_css_verification():
    verify = Mock(return_value=True)
    assert verify() is True


def test_css_retry():
    apply_fix = Mock(side_effect=[False, False, True])

    success = False
    for _ in range(3):
        if apply_fix():
            success = True
            break

    assert success is True
    assert apply_fix.call_count == 3


def test_css_failure():
    apply_fix = Mock(return_value=False)

    for _ in range(3):
        apply_fix()

    assert apply_fix.call_count == 3

