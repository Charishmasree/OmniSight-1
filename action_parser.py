"""
OmniSight - Member 3: Action Engine using Python + JSON
Action Parser Module (action_engine/action_parser.py)

Responsible for processing AI/VLM responses from Member 2:
- Safely parses AI responses (dict, string, markdown-wrapped JSON).
- Extracts and normalizes UI issues and suggested CSS fixes.
- Validates data types, required fields, and severity levels.
- Formats output into validated, structured JSON for Member 4's FastAPI backend.
- Strictly adheres to security rules: NEVER executes AI-generated CSS or code.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

logger = logging.getLogger("omnisight.action_engine.action_parser")

# ============================================================================
# Taxonomy & Constants
# ============================================================================

VALID_SEVERITIES = {"critical", "high", "medium", "low"}
DEFAULT_SEVERITY = "medium"

ISSUE_TYPE_ALIASES = {
    "misaligned_buttons": "misalignment",
    "misaligned_button": "misalignment",
    "button_misalignment": "misalignment",
    "alignment": "misalignment",
    "misaligned": "misalignment",
    "misalignment": "misalignment",
    "overlapping_elements": "overlapping_text",
    "text_overlap": "overlapping_text",
    "overlap": "overlapping_text",
    "overlapping_text": "overlapping_text",
    "clipped_text": "clipped_elements",
    "text_clipped": "clipped_elements",
    "overflow_clipping": "clipped_elements",
    "clipped_elements": "clipped_elements",
    "layout_issue": "responsive_layout",
    "layout_responsiveness": "responsive_layout",
    "responsive_layout_problems": "responsive_layout",
    "horizontal_overflow": "responsive_layout",
    "responsive_layout": "responsive_layout",
    "low_contrast": "contrast_issue",
    "unreadable_text": "contrast_issue",
    "contrast_issue": "contrast_issue",
    "broken_image": "missing_elements",
    "missing_elements": "missing_elements",
}


# ============================================================================
# Validation Result Helper
# ============================================================================

class ValidationResult(tuple):
    """
    Validation result container that behaves both as a boolean and a tuple (is_valid, errors).
    Allows:
        is_valid, errors = validate_issue(issue)
        if validate_issue(issue): ...
        self.assertTrue(validate_issue(issue))
        self.assertFalse(validate_issue(issue))
        result.is_valid
        result.errors
    """

    def __new__(cls, is_valid: bool, errors: Optional[List[str]] = None):
        err_list = list(errors) if errors else []
        return super().__new__(cls, (bool(is_valid), err_list))

    @property
    def is_valid(self) -> bool:
        return self[0]

    @property
    def errors(self) -> List[str]:
        return self[1]

    def __bool__(self) -> bool:
        return self[0]

    def __repr__(self) -> str:
        return f"ValidationResult(is_valid={self[0]}, errors={self[1]})"


# ============================================================================
# Normalization Helpers
# ============================================================================

def normalize_severity(severity: Any) -> str:
    """
    Normalizes severity string to one of: 'critical', 'high', 'medium', 'low'.
    Defaults to 'medium' for missing, non-string, or unrecognized values.
    """
    if not isinstance(severity, str):
        return DEFAULT_SEVERITY
    cleaned = severity.lower().strip()
    return cleaned if cleaned in VALID_SEVERITIES else DEFAULT_SEVERITY


def normalize_issue_type(raw_type: Any) -> str:
    """
    Normalizes issue category / type to standard taxonomy.
    """
    if not isinstance(raw_type, str) or not raw_type.strip():
        return "general_issue"
    cleaned = raw_type.lower().strip().replace("-", "_").replace(" ", "_")
    return ISSUE_TYPE_ALIASES.get(cleaned, cleaned)


def _extract_css_fix(fix_data: Any, fallback_recommendation: Optional[str] = None) -> str:
    """
    Safely extracts CSS rule or property string from suggested_fix or recommendation.
    NOTE: This extracts CSS as inert text; it is NEVER executed or evaluated.
    """
    if isinstance(fix_data, dict):
        if fix_data.get("css"):
            return str(fix_data["css"]).strip()
        if fix_data.get("css_code"):
            return str(fix_data["css_code"]).strip()
        if fix_data.get("css_rule"):
            return str(fix_data["css_rule"]).strip()
        if isinstance(fix_data.get("css_properties"), dict):
            props = fix_data["css_properties"]
            return " ".join(f"{k}: {v};" for k, v in props.items())
        if fix_data.get("explanation"):
            return str(fix_data["explanation"]).strip()
    elif isinstance(fix_data, str) and fix_data.strip():
        return fix_data.strip()

    if fallback_recommendation and isinstance(fallback_recommendation, str):
        rec = fallback_recommendation.strip()
        # Look for code snippet in backticks e.g. `margin-bottom: 20px;`
        code_match = re.search(r"`([^`]+)`", rec)
        if code_match:
            candidate = code_match.group(1).strip()
            if ":" in candidate:
                return candidate if candidate.endswith(";") else candidate + ";"
        return rec

    return ""


# ============================================================================
# Reusable Core Functions
# ============================================================================

def validate_issue(issue: Any) -> ValidationResult:
    """
    Validates a single UI issue dictionary.
    Checks:
    - Data type (must be dict)
    - Required fields: 'element', 'issue_type', 'severity', 'description', 'suggested_fix'
    - Value types and valid ranges (e.g. severity in critical/high/medium/low)

    Returns:
        ValidationResult(is_valid: bool, errors: List[str])
    """
    errors: List[str] = []

    if not isinstance(issue, dict):
        return ValidationResult(False, [f"Issue must be a dictionary, got {type(issue).__name__}."])

    # 1. Validate 'element'
    element_val = issue.get("element")
    if element_val is None:
        # Check if alternative key exists (e.g. selector, location, target)
        if not any(issue.get(k) for k in ("selector", "location", "target", "title")):
            errors.append("Missing required field: 'element'.")
    elif not isinstance(element_val, str):
        errors.append(f"Field 'element' must be a string, got {type(element_val).__name__}.")
    elif not element_val.strip():
        errors.append("Field 'element' cannot be empty.")

    # 2. Validate 'issue_type'
    type_val = issue.get("issue_type")
    if type_val is None:
        if not any(issue.get(k) for k in ("category", "type", "defect_type")):
            errors.append("Missing required field: 'issue_type'.")
    elif not isinstance(type_val, str):
        errors.append(f"Field 'issue_type' must be a string, got {type(type_val).__name__}.")
    elif not type_val.strip():
        errors.append("Field 'issue_type' cannot be empty.")

    # 3. Validate 'severity'
    sev_val = issue.get("severity")
    if sev_val is None:
        errors.append("Missing required field: 'severity'.")
    elif not isinstance(sev_val, str):
        errors.append(f"Field 'severity' must be a string, got {type(sev_val).__name__}.")
    elif sev_val.lower().strip() not in VALID_SEVERITIES:
        errors.append(
            f"Invalid severity value '{sev_val}'. Must be one of: 'critical', 'high', 'medium', 'low'."
        )

    # 4. Validate 'description'
    desc_val = issue.get("description")
    if desc_val is None:
        if not any(issue.get(k) for k in ("title", "details", "explanation")):
            errors.append("Missing required field: 'description'.")
    elif not isinstance(desc_val, str):
        errors.append(f"Field 'description' must be a string, got {type(desc_val).__name__}.")
    elif not desc_val.strip():
        errors.append("Field 'description' cannot be empty.")

    # 5. Validate 'suggested_fix'
    fix_val = issue.get("suggested_fix")
    rec_val = issue.get("recommendation")
    if fix_val is None and rec_val is None:
        errors.append("Missing required field: 'suggested_fix'.")
    elif fix_val is not None:
        if not isinstance(fix_val, (dict, str)):
            errors.append(f"Field 'suggested_fix' must be a dict or string, got {type(fix_val).__name__}.")
        elif isinstance(fix_val, dict) and "css" in fix_val:
            if not isinstance(fix_val["css"], str):
                errors.append(f"Field 'suggested_fix.css' must be a string, got {type(fix_val['css']).__name__}.")

    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors)


def normalize_issue(issue: Any) -> Dict[str, Any]:
    """
    Converts a single UI issue dictionary into the standard OmniSight format:
    {
      "element": str,
      "issue_type": str,
      "severity": str,
      "description": str,
      "suggested_fix": {
        "css": str
      }
    }

    Safely recovers missing or alternative fields using resilient defaults.
    """
    if not isinstance(issue, dict):
        return {
            "element": "Unknown element",
            "issue_type": "general_issue",
            "severity": DEFAULT_SEVERITY,
            "description": "Malformed or non-dictionary issue record.",
            "suggested_fix": {"css": ""},
        }

    # Extract element
    element = (
        issue.get("element")
        or issue.get("selector")
        or issue.get("location")
        or issue.get("target")
        or issue.get("title")
        or "Unknown element"
    )
    element = str(element).strip() if element else "Unknown element"

    # Extract issue type
    raw_type = (
        issue.get("issue_type")
        or issue.get("category")
        or issue.get("type")
        or issue.get("defect_type")
        or "general_issue"
    )
    issue_type = normalize_issue_type(raw_type)

    # Extract severity
    raw_sev = issue.get("severity")
    severity = normalize_severity(raw_sev)

    # Extract description
    description = (
        issue.get("description")
        or issue.get("title")
        or issue.get("details")
        or issue.get("explanation")
        or "No description provided."
    )
    description = str(description).strip() if description else "No description provided."

    # Extract suggested CSS fix
    css_fix_str = _extract_css_fix(
        fix_data=issue.get("suggested_fix"),
        fallback_recommendation=issue.get("recommendation"),
    )

    return {
        "element": element,
        "issue_type": issue_type,
        "severity": severity,
        "description": description,
        "suggested_fix": {
            "css": css_fix_str,
        },
    }


def _safe_load_json_string(text: str) -> Optional[Any]:
    """
    Attempts multiple parsing strategies to parse a JSON string:
    1. Direct json.loads
    2. Stripping markdown fences (```json ... ```)
    3. Finding outermost curly braces { ... } or brackets [ ... ]
    """
    cleaned = text.strip()

    # 1. Strip markdown fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    # Direct parse attempt
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # Extract substring between first '{' and last '}'
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = cleaned[first_brace : last_brace + 1]
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass

    # Extract substring between first '[' and last ']'
    first_bracket = cleaned.find("[")
    last_bracket = cleaned.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        candidate = cleaned[first_bracket : last_bracket + 1]
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def parse_ai_response(ai_response: Any) -> Dict[str, Any]:
    """
    Accepts Member 2's AI/VLM response, parses and validates it safely,
    normalizes detected UI issues, and returns a structured dictionary:
    {
      "status": "success" | "error",
      "total_issues": int,
      "issues": [...],
      "errors": [...]
    }

    Handles:
    - JSON strings (plain or markdown-wrapped)
    - Empty or None responses
    - Malformed JSON strings
    - Unexpected data types
    - Missing required fields
    - Invalid severity values
    - Zero issues (pristine page)
    - Automatically guarantees total_issues == len(issues)
    """
    # -------------------------------------------------------------------------
    # 1. Check for empty, None, or invalid top-level types
    # -------------------------------------------------------------------------
    if ai_response is None:
        return {
            "status": "error",
            "total_issues": 0,
            "issues": [],
            "errors": ["AI response is None."],
        }

    if isinstance(ai_response, str):
        if not ai_response.strip():
            return {
                "status": "error",
                "total_issues": 0,
                "issues": [],
                "errors": ["AI response string is empty."],
            }
        parsed = _safe_load_json_string(ai_response)
        if parsed is None:
            return {
                "status": "error",
                "total_issues": 0,
                "issues": [],
                "errors": [f"Malformed JSON in AI response string: {ai_response[:100]}..."],
            }
        raw_data = parsed
    elif isinstance(ai_response, (dict, list)):
        raw_data = ai_response
    else:
        return {
            "status": "error",
            "total_issues": 0,
            "issues": [],
            "errors": [
                f"Unexpected data type for AI response: '{type(ai_response).__name__}'. Expected dict, str, or list."
            ],
        }

    # If raw_data is an empty dictionary
    if isinstance(raw_data, dict) and len(raw_data) == 0:
        return {
            "status": "error",
            "total_issues": 0,
            "issues": [],
            "errors": ["AI response dictionary is empty."],
        }

    # -------------------------------------------------------------------------
    # 2. Extract raw issues list
    # -------------------------------------------------------------------------
    raw_issues: List[Any]
    top_level_errors: List[str] = []

    if isinstance(raw_data, list):
        raw_issues = raw_data
    elif isinstance(raw_data, dict):
        if "issues" in raw_data and raw_data["issues"] is not None:
            val = raw_data["issues"]
            if isinstance(val, list):
                raw_issues = val
            else:
                top_level_errors.append(f"Expected 'issues' to be a list, got {type(val).__name__}.")
                raw_issues = []
        elif "visual_issues" in raw_data and raw_data["visual_issues"] is not None:
            val = raw_data["visual_issues"]
            if isinstance(val, list):
                raw_issues = val
            else:
                top_level_errors.append(f"Expected 'visual_issues' to be a list, got {type(val).__name__}.")
                raw_issues = []
        # Check if the dictionary itself is a single issue object
        elif any(k in raw_data for k in ("element", "issue_type", "severity", "category", "selector")):
            raw_issues = [raw_data]
        else:
            top_level_errors.append("Missing 'issues' field in AI response.")
            raw_issues = []
    else:
        raw_issues = []

    # -------------------------------------------------------------------------
    # 3. Handle zero issues cleanly (clean/pristine UI state)
    # -------------------------------------------------------------------------
    if len(raw_issues) == 0:
        status = "error" if top_level_errors else "success"
        return {
            "status": status,
            "total_issues": 0,
            "issues": [],
            "errors": top_level_errors,
        }

    # -------------------------------------------------------------------------
    # 4. Validate and normalize each issue
    # -------------------------------------------------------------------------
    normalized_issues: List[Dict[str, Any]] = []
    issue_errors: List[str] = list(top_level_errors)

    for idx, item in enumerate(raw_issues):
        if not isinstance(item, dict):
            issue_errors.append(f"Issue at index {idx} has unexpected type '{type(item).__name__}', expected dict.")
            continue

        # Run validation check
        val_result = validate_issue(item)
        if not val_result.is_valid:
            issue_name = item.get("element") or item.get("selector") or f"index {idx}"
            for err in val_result.errors:
                issue_errors.append(f"Issue [{issue_name}]: {err}")

        # Normalize issue into standard format with resilient defaults
        normalized = normalize_issue(item)
        normalized_issues.append(normalized)

    # Status is 'error' if any validation or top-level errors occurred, otherwise 'success'
    status = "error" if len(issue_errors) > 0 else "success"

    return {
        "status": status,
        "total_issues": len(normalized_issues),
        "issues": normalized_issues,
        "errors": issue_errors,
    }


def convert_to_json(result: Any, indent: int = 2) -> str:
    """
    Serializes the final result dictionary into a valid JSON string.
    Ensures safe formatting for Member 4's FastAPI integration.
    """
    if isinstance(result, str):
        # Check if it's already a valid JSON string
        try:
            parsed = json.loads(result)
            return json.dumps(parsed, indent=indent)
        except Exception:
            return json.dumps({"raw_output": result}, indent=indent)

    try:
        return json.dumps(result, indent=indent)
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to serialize result to JSON: {e}")
        return json.dumps({
            "status": "error",
            "total_issues": 0,
            "issues": [],
            "errors": [f"JSON serialization error: {str(e)}"],
        }, indent=indent)


# ============================================================================
# Pydantic Schemas for Member 4 FastAPI Integration
# ============================================================================

class SuggestedFixModel(BaseModel):
    """Suggested CSS fix model."""
    css: str = Field(default="", description="Inert suggested CSS rule or properties")


class IssueModel(BaseModel):
    """Validated UI defect representation."""
    element: str = Field(default="Unknown element", description="Target UI element")
    issue_type: str = Field(default="general_issue", description="Standardized issue category")
    severity: str = Field(default="medium", description="Issue severity: critical, high, medium, low")
    description: str = Field(default="", description="Human-readable defect description")
    suggested_fix: SuggestedFixModel = Field(default_factory=SuggestedFixModel)


class ActionEngineResponseModel(BaseModel):
    """Complete Action Engine response schema for FastAPI."""
    status: str = Field(default="success", description="Status: 'success' or 'error'")
    total_issues: int = Field(default=0, description="Total count of issues returned")
    issues: List[IssueModel] = Field(default_factory=list, description="List of detected UI issues")
    errors: List[str] = Field(default_factory=list, description="List of validation errors or warnings")
