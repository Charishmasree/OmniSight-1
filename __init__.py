"""
OmniSight - Member 3: Action Engine using Python + JSON
Action Engine Package (action_engine)

Validates Member 2's AI/VLM responses, extracts visual issues and CSS fixes,
and returns standardized, JSON-compatible output ready for Member 4's FastAPI backend.
"""

from action_engine.action_parser import (
    parse_ai_response,
    normalize_issue,
    validate_issue,
    convert_to_json,
    ValidationResult,
    IssueModel,
    ActionEngineResponseModel,
    SuggestedFixModel as ParserSuggestedFixModel,
    VALID_SEVERITIES,
    DEFAULT_SEVERITY,
)
from action_engine.processor import process_ai_response
from action_engine.validators import (
    validate_ai_response,
    CSSFixModel,
    VisualIssueModel,
    SuggestedFixModel,
    StandardizedReportModel,
)

__all__ = [
    "parse_ai_response",
    "normalize_issue",
    "validate_issue",
    "convert_to_json",
    "ValidationResult",
    "IssueModel",
    "ActionEngineResponseModel",
    "ParserSuggestedFixModel",
    "VALID_SEVERITIES",
    "DEFAULT_SEVERITY",
    "process_ai_response",
    "validate_ai_response",
    "CSSFixModel",
    "VisualIssueModel",
    "SuggestedFixModel",
    "StandardizedReportModel",
]
