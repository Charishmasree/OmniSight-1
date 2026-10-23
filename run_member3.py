"""
OmniSight - Member 3: Action Engine Runner & Demonstration Script
Run this script directly in VS Code (F5, 'Run Python File', or via Terminal).

Demonstrates Member 3's core responsibilities:
1. Parsing & validating AI/VLM output from Member 2
2. Normalizing UI defect schemas & extracting inert CSS fixes
3. Calculating visual health scores & prioritizing remediation actions
4. Exporting 100% JSON-compliant payload for Member 4's FastAPI integration
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from action_engine.action_parser import parse_ai_response, convert_to_json
from action_engine.processor import process_ai_response


def print_banner(title: str):
    width = 70
    print("\n" + "=" * width)
    print(f" {title.center(width - 2)} ")
    print("=" * width)


def run_member3_demo():
    print_banner("OmniSight - Member 3: Action Engine Demonstration")

    sample_input_path = PROJECT_ROOT / "analysis_reports" / "example_member2_ai_output.json"
    if sample_input_path.exists():
        print(f"\n[1] Loading Member 2 AI Output from:\n    {sample_input_path.name}")
        with open(sample_input_path, "r", encoding="utf-8") as f:
            member2_data = json.load(f)
    else:
        print("\n[1] Using built-in sample AI response dictionary...")
        member2_data = {
            "page_summary": "Dashboard UI with navigation bar and action button.",
            "visual_health_score": 75,
            "issues": [
                {
                    "issue_id": "ISSUE-001",
                    "category": "overlapping_text",
                    "severity": "high",
                    "selector": ".hero-subtitle",
                    "description": "Subtitle text collides with primary CTA button.",
                    "suggested_fix": {
                        "selector": ".hero-subtitle",
                        "css_properties": {"margin-bottom": "24px"},
                        "css_code": ".hero-subtitle { margin-bottom: 24px; }",
                        "explanation": "Add bottom margin to push the button below the text block.",
                    },
                },
                {
                    "issue_id": "ISSUE-002",
                    "category": "misaligned_buttons",
                    "severity": "medium",
                    "selector": ".btn-group .btn-secondary",
                    "description": "Secondary button is offset by 6px relative to primary button.",
                    "suggested_fix": {
                        "selector": ".btn-group",
                        "css_properties": {"display": "inline-flex", "align-items": "center"},
                        "css_code": ".btn-group { display: inline-flex; align-items: center; }",
                        "explanation": "Align buttons using inline flexbox.",
                    },
                },
            ],
        }

    # Step 2: Run Action Parser
    print_banner("Step 2: Action Engine Parsing & Normalization")
    parsed_report = parse_ai_response(member2_data)
    print(f"  • Status:       {parsed_report['status']}")
    print(f"  • Total Issues: {parsed_report['total_issues']}")
    print(f"  • Errors Found: {len(parsed_report['errors'])}")

    for i, issue in enumerate(parsed_report["issues"], 1):
        print(f"\n  [Issue #{i}]")
        print(f"    - Element:       {issue.get('element')}")
        print(f"    - Type/Category: {issue.get('issue_type')}")
        print(f"    - Severity:      {issue.get('severity')}")
        print(f"    - Description:   {issue.get('description')}")
        fix = issue.get("suggested_fix", {}).get("css", "N/A")
        print(f"    - Inert CSS Fix: {fix}")

    # Step 3: Run Full Processor (prioritization, health score, Member 4 FastAPI payload)
    print_banner("Step 3: Standardized Action Engine Payload (Member 4 Ready)")
    processed = process_ai_response(member2_data)
    print(f"  • Health Score:       {processed.get('visual_health_score')}/100")
    print(f"  • Total Detected:     {processed.get('total_issues')}")
    print(f"  • Severity Breakdown: {json.dumps(processed.get('issues_by_severity', {}))}")
    print(f"  • Prioritized Actions: {len(processed.get('actions', []))} action(s) compiled")

    for act in processed.get("actions", []):
        print(f"    - [Priority {act.get('priority')}] ({act.get('severity').upper()}) {act.get('selector')}: {act.get('explanation')}")

    # Step 4: Verify JSON serialization
    json_output = convert_to_json(processed)
    output_demo_file = PROJECT_ROOT / "analysis_reports" / "member3_run_output.json"
    with open(output_demo_file, "w", encoding="utf-8") as f:
        f.write(json_output)
    print(f"\n[4] Standardized JSON successfully generated and saved to:\n    {output_demo_file.name}")

    print_banner("SUCCESS: Member 3 Action Engine Execution Complete!")
    print("\nTo run the full unit test suite for Member 3, execute:\n  python action_engine/test_action_parser.py\n  python action_engine/test_processor.py\n")


if __name__ == "__main__":
    run_member3_demo()
