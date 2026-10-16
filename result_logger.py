"""
OmniSight - Member 2: AI/VLM Basic Setup and Prompt Testing
Result Logger Module (result_logger.py)

Handles serializing analysis results to JSON files, saving reports to disk,
loading saved reports, and printing clean terminal summaries.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

from config import get_config

logger = logging.getLogger("omnisight.result_logger")


def generate_default_filename(screenshot_path: Optional[str] = None) -> str:
    """
    Generates a timestamped default filename based on the screenshot name.
    Example: 'dashboard_analysis_20261016_070000.json'
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if screenshot_path:
        stem = Path(screenshot_path).stem
        # Clean up any non-alphanumeric characters for a safe filename
        safe_stem = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stem)
        return f"{safe_stem}_analysis_{timestamp}.json"
    return f"omnisight_analysis_{timestamp}.json"


def save_analysis_report(
    data: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None
) -> Path:
    """
    Saves the structured analysis dictionary to a pretty-formatted JSON file.

    Args:
        data: The dictionary returned by VisionAnalyzer.analyze_screenshot.
        output_path: Exact full path for the output file (overrides output_dir and filename).
        output_dir: Target directory (defaults to config's output_dir if not specified).
        filename: Optional custom filename (e.g. 'report.json').

    Returns:
        Path to the saved JSON file.

    Raises:
        TypeError: If data is not serializable to JSON.
        OSError: If directory creation or file writing fails.
    """
    config = get_config()

    if output_path:
        target_file = Path(output_path)
    else:
        target_dir = Path(output_dir) if output_dir else config.output_dir
        if not filename:
            # Extract screenshot path from metadata if present
            screenshot_path = data.get("metadata", {}).get("screenshot_path")
            filename = generate_default_filename(screenshot_path)
        target_file = target_dir / filename

    # Ensure parent directories exist
    target_file.parent.mkdir(parents=True, exist_ok=True)

    # Write serialized JSON
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Analysis report successfully saved to: {target_file}")
    return target_file


def load_analysis_report(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads and parses an existing JSON analysis report from disk.

    Args:
        file_path: Path to the JSON report file.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Report file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_summary_report(data: Dict[str, Any]) -> str:
    """
    Builds a clean, human-readable terminal summary of the analysis report.
    Useful for developers, test logs, and CLI verification.
    """
    meta = data.get("metadata", {})
    page_summary = data.get("page_summary", "No summary provided.")
    health_score = data.get("visual_health_score", "N/A")
    ui_elements = data.get("ui_elements", {})
    issues = data.get("visual_issues", [])
    recommendations = data.get("recommendations", [])

    # Count elements
    button_count = len(ui_elements.get("buttons", []))
    input_count = len(ui_elements.get("input_fields", []))
    text_count = len(ui_elements.get("text_elements", []))
    nav_count = len(ui_elements.get("navigation_elements", []))
    media_count = len(ui_elements.get("media_elements", []))

    # Issue counts by severity
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for issue in issues:
        sev = str(issue.get("severity", "medium")).lower()
        if sev in severity_counts:
            severity_counts[sev] += 1
        else:
            severity_counts["medium"] += 1

    divider = "=" * 70
    subdivider = "-" * 70

    lines = [
        divider,
        "  OMNISIGHT UI/UX VISION ANALYSIS REPORT",
        divider,
        f"Screenshot:      {meta.get('screenshot_path', 'N/A')}",
        f"Model:           {meta.get('model', 'N/A')}",
        f"Analysis Time:   {meta.get('timestamp', 'N/A')}",
        f"Execution Time:  {meta.get('duration_seconds', 'N/A')}s",
        subdivider,
        f"Page Summary:    {page_summary}",
        f"Health Score:    {health_score}/100",
        subdivider,
        "UI Elements Detected:",
        f"  - Buttons:            {button_count}",
        f"  - Input Fields:       {input_count}",
        f"  - Text Elements:      {text_count}",
        f"  - Navigation Items:   {nav_count}",
        f"  - Media/Images:       {media_count}",
        subdivider,
        f"Visual Issues Detected: {len(issues)} (Critical: {severity_counts['critical']}, High: {severity_counts['high']}, Med: {severity_counts['medium']}, Low: {severity_counts['low']})",
    ]

    if issues:
        for idx, issue in enumerate(issues, 1):
            lines.append(f"\n  [{idx}] {issue.get('title', 'Untitled Issue')}")
            lines.append(f"      Category:       {issue.get('category', 'general')}")
            lines.append(f"      Severity:       {issue.get('severity', 'medium').upper()}")
            lines.append(f"      Location:       {issue.get('location', 'unspecified')}")
            lines.append(f"      Description:    {issue.get('description', '')}")
            lines.append(f"      Recommendation: {issue.get('recommendation', 'N/A')}")
    else:
        lines.append("  No visual issues detected. Clean UI presentation!")

    if recommendations:
        lines.append(subdivider)
        lines.append("General Recommendations:")
        for rec in recommendations:
            lines.append(f"  * {rec}")

    lines.append(divider)
    return "\n".join(lines)


def print_summary(data: Dict[str, Any]) -> None:
    """Prints the human-readable summary to standard output."""
    print(format_summary_report(data))
