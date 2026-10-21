"""
OmniSight - Member 3: Action Engine using Python + JSON
Processor Module (action_engine/processor.py)

Processes Member 2's AI/VLM responses:
- Validates structure and handles malformed JSON safely
- Extracts and normalizes visual issues and CSS fixes
- Generates prioritized remediation actions
- Computes issue severity and category statistics
- Returns standardized, 100% JSON-compatible output for Member 4 (FastAPI)
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union, List

from action_engine.validators import (
    normalize_category,
    normalize_severity,
    parse_css_properties_from_string,
    validate_ai_response,
    SEVERITY_WEIGHTS,
)

logger = logging.getLogger("omnisight.action_engine.processor")


def _extract_json_string(text: str) -> Optional[Dict[str, Any]]:
    """
    Attempts multiple parsing strategies to parse a JSON string from AI output:
    1. Direct json.loads
    2. Stripping markdown fences (```json ... ```)
    3. Finding the outermost curly braces { ... }
    """
    cleaned = text.strip()

    # 1. Strip markdown fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    # Try direct parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Extract substring between first '{' and last '}'
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = cleaned[first_brace : last_brace + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _calculate_health_score(issues: List[Dict[str, Any]], provided_score: Optional[int] = None) -> int:
    """
    Calculates or validates visual health score between 0 and 100.
    Deducts penalty based on severity if score was missing.
    """
    if provided_score is not None and isinstance(provided_score, (int, float)):
        return max(0, min(100, int(provided_score)))

    if not issues:
        return 100

    deductions = {
        "critical": 30,
        "high": 18,
        "medium": 10,
        "low": 4,
    }
    score = 100
    for issue in issues:
        sev = normalize_severity(issue.get("severity"))
        score -= deductions.get(sev, 10)

    return max(0, min(100, score))


def _build_error_response(error_message: str, raw_input: Any = None) -> Dict[str, Any]:
    """Builds a standardized error response dictionary."""
    return {
        "status": "error",
        "valid": False,
        "page_url": None,
        "page_summary": "Failed to parse or validate AI response.",
        "visual_health_score": 0,
        "total_issues": 0,
        "issues_by_severity": {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        },
        "issues_by_category": {},
        "issues": [],
        "css_fixes": {},
        "actions": [],
        "ui_elements": {},
        "recommendations": [],
        "metadata": {
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "engine": "OmniSight Action Engine v2.0",
            "source_status": "error",
            "error_type": "ValidationError",
        },
        "errors": [error_message],
    }


def process_ai_response(
    ai_response: Union[str, Dict[str, Any], None]
) -> Dict[str, Any]:
    """
    Validates Member 2's AI response, extracts issues and CSS fixes,
    and returns a standardized JSON-compatible dictionary.

    Handles:
    - Invalid JSON strings safely (no uncaught exceptions)
    - Missing fields with resilient defaults
    - Empty issues lists cleanly
    - Legacy / alternate key names (e.g. visual_issues vs issues)

    Args:
        ai_response: Raw dictionary from Member 2, or raw JSON string, or None.

    Returns:
        Standardized, 100% JSON-compatible dictionary with:
        - status: "success" | "warning" | "error"
        - valid: bool
        - page_url: str | None
        - page_summary: str
        - visual_health_score: int (0-100)
        - total_issues: int
        - issues_by_severity: dict
        - issues_by_category: dict
        - issues: list of standardized issue dicts with structured suggested_fix
        - css_fixes: structured dictionary mapping selectors to CSS rules
        - actions: prioritized list of remediation actions
        - ui_elements: dict of identified UI elements
        - recommendations: list of strings
        - metadata: processing timestamp and version info
        - errors: list of warnings/errors encountered
    """
    # -------------------------------------------------------------------------
    # 1. Safe Input Parsing
    # -------------------------------------------------------------------------
    if ai_response is None:
        return _build_error_response("AI response is None.")

    raw_data: Dict[str, Any]
    if isinstance(ai_response, str):
        if not ai_response.strip():
            return _build_error_response("AI response string is empty.")
        parsed = _extract_json_string(ai_response)
        if parsed is None:
            return _build_error_response(
                f"Invalid JSON in AI response string: {ai_response[:100]}..."
            )
        raw_data = parsed
    elif isinstance(ai_response, dict):
        raw_data = dict(ai_response)
    else:
        return _build_error_response(
            f"Unsupported AI response type '{type(ai_response).__name__}'. Expected dict or JSON string."
        )

    # -------------------------------------------------------------------------
    # 2. Validation & Missing Fields Recovery
    # -------------------------------------------------------------------------
    is_valid, sanitized, validation_errors = validate_ai_response(raw_data)

    page_url = sanitized.get("page_url")
    page_summary = sanitized.get("page_summary") or "Visual layout analysis completed."
    recommendations = sanitized.get("recommendations") or []
    if not isinstance(recommendations, list):
        recommendations = [str(recommendations)]
    ui_elements = sanitized.get("ui_elements") or {}
    if not isinstance(ui_elements, dict):
        ui_elements = {}

    # Read raw issues list (support both 'issues' and 'visual_issues')
    raw_issues = sanitized.get("issues") or sanitized.get("visual_issues") or []
    if not isinstance(raw_issues, list):
        raw_issues = []

    # -------------------------------------------------------------------------
    # 3. Extraction & Normalization of Issues & CSS Fixes
    # -------------------------------------------------------------------------
    standardized_issues: List[Dict[str, Any]] = []
    css_fixes: Dict[str, Any] = {}
    actions: List[Dict[str, Any]] = []

    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    category_counts: Dict[str, int] = {}

    for idx, item in enumerate(raw_issues):
        if not isinstance(item, dict):
            continue

        issue_id = str(item.get("issue_id") or f"ISSUE-{idx + 1:03d}")
        category = normalize_category(item.get("category"))
        severity = normalize_severity(item.get("severity"))
        title = str(item.get("title") or f"Visual defect ({category})")
        location = str(item.get("location") or "Screen layout")
        selector = str(item.get("selector") or "").strip()
        description = str(item.get("description") or title)

        # Count statistics
        severity_counts[severity] += 1
        category_counts[category] = category_counts.get(category, 0) + 1

        # Extract or build structured suggested CSS fix
        suggested = item.get("suggested_fix")
        suggested_dict: Dict[str, Any] = {}

        if isinstance(suggested, dict):
            fix_sel = str(suggested.get("selector") or selector or f"#{issue_id.lower()}").strip()
            props = suggested.get("css_properties") or {}
            if not isinstance(props, dict):
                props = parse_css_properties_from_string(str(props))
            # Clean string values
            clean_props = {str(k).strip(): str(v).strip() for k, v in props.items()}

            css_code = str(suggested.get("css_code") or "").strip()
            if not css_code and fix_sel and clean_props:
                prop_str = " ".join(f"{k}: {v};" for k, v in clean_props.items())
                css_code = f"{fix_sel} {{ {prop_str} }}"

            explanation = str(suggested.get("explanation") or item.get("recommendation") or title).strip()

            suggested_dict = {
                "selector": fix_sel,
                "css_properties": clean_props,
                "css_code": css_code,
                "explanation": explanation,
            }
        else:
            # Fallback if suggested_fix was missing or provided as a legacy string recommendation
            rec_text = str(item.get("recommendation") or suggested or "").strip()
            inferred_props = parse_css_properties_from_string(rec_text)
            fix_sel = selector or f"#{issue_id.lower()}"
            prop_str = " ".join(f"{k}: {v};" for k, v in inferred_props.items())
            css_code = f"{fix_sel} {{ {prop_str} }}" if (fix_sel and prop_str) else rec_text

            suggested_dict = {
                "selector": fix_sel,
                "css_properties": inferred_props,
                "css_code": css_code,
                "explanation": rec_text or f"Fix for {title}",
            }

        # If issue selector was missing, backfill from fix_sel
        if not selector and suggested_dict.get("selector"):
            selector = suggested_dict["selector"]

        issue_obj = {
            "issue_id": issue_id,
            "category": category,
            "severity": severity,
            "title": title,
            "location": location,
            "selector": selector,
            "description": description,
            "suggested_fix": suggested_dict,
        }
        standardized_issues.append(issue_obj)

        # Populate top-level CSS fixes
        if suggested_dict.get("selector") and (suggested_dict.get("css_properties") or suggested_dict.get("css_code")):
            sel_key = suggested_dict["selector"]
            css_fixes[sel_key] = {
                "selector": sel_key,
                "properties": suggested_dict["css_properties"],
                "css_rule": suggested_dict["css_code"],
                "issue_id": issue_id,
                "description": suggested_dict["explanation"],
            }

        # Build prioritized action item
        actions.append({
            "priority": SEVERITY_WEIGHTS.get(severity, 3),
            "issue_id": issue_id,
            "severity": severity,
            "category": category,
            "selector": selector,
            "action_type": "apply_css_fix",
            "css_rule": suggested_dict.get("css_code", ""),
            "explanation": suggested_dict.get("explanation", ""),
        })

    # Sort actions by priority (critical first, then high, medium, low)
    actions.sort(key=lambda x: x["priority"])

    # Incorporate pre-existing css_fixes from raw_data if any
    raw_css_fixes = sanitized.get("css_fixes")
    if isinstance(raw_css_fixes, dict):
        for k, v in raw_css_fixes.items():
            if k not in css_fixes and isinstance(v, dict):
                css_fixes[k] = {
                    "selector": str(v.get("selector") or k),
                    "properties": v.get("properties") or {},
                    "css_rule": str(v.get("css_rule") or ""),
                    "issue_id": str(v.get("issue_id") or ""),
                    "description": str(v.get("description") or ""),
                }

    # -------------------------------------------------------------------------
    # 4. Empty Issues Handling
    # -------------------------------------------------------------------------
    total_issues = len(standardized_issues)
    if total_issues == 0:
        if not recommendations:
            recommendations = ["No remediation actions needed. Page visual styling and layout are healthy."]
        if page_summary == "Visual layout analysis completed.":
            page_summary = "All visual quality checks passed. No layout defects or overlaps detected."

    # Compute or clamp health score
    visual_health_score = _calculate_health_score(
        issues=standardized_issues,
        provided_score=sanitized.get("visual_health_score"),
    )

    # Status determination: "success" if valid, "error" if invalid
    status = "success" if is_valid else "error"

    # -------------------------------------------------------------------------
    # 5. Build Final Standardized Output
    # -------------------------------------------------------------------------
    output = {
        "status": status,
        "valid": is_valid,
        "page_url": page_url,
        "page_summary": page_summary,
        "visual_health_score": visual_health_score,
        "total_issues": total_issues,
        "issues_by_severity": severity_counts,
        "issues_by_category": category_counts,
        "issues": standardized_issues,
        "css_fixes": css_fixes,
        "actions": actions,
        "ui_elements": ui_elements,
        "recommendations": recommendations,
        "metadata": {
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "engine": "OmniSight Action Engine v2.0",
            "source_status": sanitized.get("status") or "processed",
            "source_model": (sanitized.get("metadata") or {}).get("model", "unknown"),
        },
        "errors": validation_errors,
    }

    return output
