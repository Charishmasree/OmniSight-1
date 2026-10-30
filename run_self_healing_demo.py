"""
OmniSight - Member 2: Week 3 Self-Healing Loop Demonstration
Run this script directly in VS Code (F5, 'Run Python File', or via Terminal).

Demonstrates:
1. Loading a local webpage with an intentional visual defect (overlapping text).
2. Extracting suggested CSS fix from VLM analysis.
3. Applying runtime CSS to the live browser DOM via Playwright.
4. Capturing post-fix screenshots.
5. Visual verification using Google Gemini VLM (or offline mock mode).
6. Generating structured JSON verification results.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.self_healing_loop import run_self_healing_loop, SelfHealingLoop
from config import get_config


def print_banner(title: str):
    width = 75
    print("\n" + "=" * width)
    print(f" {title.center(width - 2)} ")
    print("=" * width)


def main():
    parser = argparse.ArgumentParser(
        description="OmniSight Week 3 - Autonomous UI Self-Healing Loop Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--page", "-p",
        type=str,
        default=str(PROJECT_ROOT / "sample_pages" / "defect_page.html"),
        help="Path or URL of test page (default: sample_pages/defect_page.html)",
    )
    parser.add_argument(
        "--max-attempts", "-a",
        type=int,
        default=3,
        help="Maximum retry attempts (default: 3)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run against live Google Gemini API (requires GEMINI_API_KEY). By default runs in offline mock mode.",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Launch browser with visible UI (non-headless) for visual inspection.",
    )

    args = parser.parse_args()

    print_banner("OmniSight - Week 3: Multimodal UI Self-Healing Loop")

    # Step 1: Define or load the detected UI issue
    detected_issue = {
        "issue_id": "ISSUE-001",
        "category": "overlapping_text",
        "severity": "high",
        "title": "Hero subtitle collides with primary action button",
        "location": "Hero banner center",
        "selector": ".hero-subtitle",
        "description": "Negative margin causes subtitle text to collide with and obscure the top of the CTA button.",
        "suggested_fix": {
            "selector": ".hero-subtitle",
            "css_properties": {
                "margin-bottom": "24px",
                "line-height": "1.4",
            },
            "css_code": ".hero-subtitle { margin-bottom: 24px !important; line-height: 1.4 !important; }",
            "explanation": "Normalizes bottom margin to 24px and line-height to 1.4, removing overlap with the CTA button.",
        },
    }

    mock_mode = not args.live
    print(f"\n[1] Target Page:      {args.page}")
    print(f"[2] Target Defect:    {detected_issue['title']} ({detected_issue['category']})")
    print(f"[3] Target Selector:  {detected_issue['selector']}")
    print(f"[4] Execution Mode:   {'OFFLINE MOCK (Deterministic QA)' if mock_mode else 'LIVE GOOGLE GEMINI VLM'}")
    print(f"[5] Browser Headless: {not args.visible}")
    print(f"[6] Max Attempts:     {args.max_attempts}")

    print_banner("Executing Autonomous Self-Healing Loop")

    result = run_self_healing_loop(
        page_target=args.page,
        detected_issue=detected_issue,
        max_attempts=args.max_attempts,
        mock_mode=mock_mode,
        headless=not args.visible,
    )

    print_banner("Self-Healing Verification Results")
    print(f"  * Verification Status: {result.get('verification_status', '').upper()}")
    print(f"  * Success Flag:        {result.get('success')}")
    print(f"  * Attempts Completed:  {result.get('attempt')}/{result.get('max_attempts')}")
    print(f"  * Reason:              {result.get('verification_reason')}")
    print(f"  * Suggested CSS:       {result.get('suggested_css')}")
    print(f"  * Post-fix Screenshot: {result.get('screenshot_path')}")

    if result.get("history"):
        print("\n  --- Attempt Progression ---")
        for h in result["history"]:
            status_symbol = "OK" if h.get("verification_status") == "fixed" else "FAIL"
            print(f"    [{status_symbol}] Attempt {h.get('attempt')}: Status={h.get('verification_status')} - {h.get('verification_reason')[:70]}...")


    output_file = PROJECT_ROOT / "analysis_reports" / "self_healing_demo_output.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n[SUCCESS] Standardized Verification Report saved to:\n  {output_file.resolve()}")
    print_banner("Self-Healing Demonstration Complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
