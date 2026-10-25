
import json
from typing import Any


REQUIRED_FIELDS = [
    "element",
    "issue_type",
    "severity",
    "description",
    "suggested_css_fix",
]


def format_issues(issues: Any) -> dict:
    """Validate UI issues and return structured JSON data."""

    if not isinstance(issues, list):
        raise ValueError("Issues must be provided as a list.")

    formatted_issues = []

    for index, issue in enumerate(issues):
        if not isinstance(issue, dict):
            raise ValueError(
                f"Issue at index {index} must be a dictionary."
            )

        missing_fields = [
            field for field in REQUIRED_FIELDS
            if field not in issue
        ]

        if missing_fields:
            raise ValueError(
                f"Issue at index {index} is missing fields: "
                f"{', '.join(missing_fields)}"
            )

        formatted_issue = {}

        for field in REQUIRED_FIELDS:
            value = issue[field]

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"'{field}' at issue index {index} "
                    "must be a non-empty string."
                )

            formatted_issue[field] = value.strip()

        formatted_issues.append(formatted_issue)

    return {"issues": formatted_issues}


def to_json(issues: Any) -> str:
    """Convert validated issues into a formatted JSON string."""

    result = format_issues(issues)
    return json.dumps(result, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sample_issues = [
        {
            "element": "Submit button",
            "issue_type": "misalignment",
            "severity": "medium",
            "description": "The button is not aligned with the form.",
            "suggested_css_fix": "margin-left: auto;"
        }
    ]

    print(to_json(sample_issues))
