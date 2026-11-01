"""
OmniSight - Member 2: AI Integration using Vision-Language Models (VLM)
Prompts Module (ai/prompts.py)

Contains reusable system prompts, inspection prompt builders, and JSON schema definitions
for analyzing website screenshots with Google Gemini VLM.
Instructs the model to detect visual UI issues such as overlapping text, misaligned buttons,
clipped elements, and responsive layout problems, and to generate structured CSS fixes.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone


# ============================================================================
# System Instructions
# ============================================================================


SYSTEM_PROMPT = """You are OmniSight's Senior AI Visual QA Inspector.

Your task is to analyze website screenshots and identify visual UI defects.
Use the provided HTML DOM context, when available, to identify the correct
HTML elements and CSS selectors.

INSPECTION AREAS:

1. overlapping_text:
   Detect text overlapping buttons, images, cards, or other text.

2. misaligned_buttons:
   Detect buttons that are incorrectly aligned with other elements.

3. clipped_elements:
   Detect text, images, icons, or containers cut off by their boundaries.

4. responsive_layout:
   Detect horizontal overflow, broken layouts, and incorrect element wrapping.

5. unreadable_text:
   Detect text that is too small, difficult to read, or has poor contrast.

6. missing_elements:
   Detect visibly missing, broken, or incorrectly rendered UI elements.

7. other:
   Identify other clearly visible visual defects.

ANALYSIS RULES:

- Inspect the complete screenshot carefully.
- Report only defects supported by visible evidence.
- Do not invent defects or assume that an element is broken.
- Use the provided HTML to identify accurate CSS selectors.
- If HTML is unavailable, use the best reasonable selector.
- Do not invent HTML elements that are not visible.
- Assign severity as critical, high, medium, or low.
- Avoid reporting the same defect more than once.
- If no defects are found, return an empty issues array.
- Do not suggest CSS changes that could unnecessarily break
  the existing layout.
- Provide a specific explanation for every suggested fix.

CSS FIX RULES:

- Suggest valid CSS property-value pairs.
- Provide a complete CSS rule.
- Prefer minimal and targeted CSS changes.
- Do not use JavaScript as a CSS fix.
- Do not suggest removing essential page content.
- Do not claim a fix has been applied or verified.

OUTPUT RULES:

- Follow the JSON schema provided in the analysis prompt.
- Return only one valid JSON object.
- Do not include Markdown code fences.
- Do not include explanations outside the JSON object.
- Use an empty array when no defects are detected.
"""


# ============================================================================
# Output JSON Schema Specification
# ============================================================================

JSON_SCHEMA_INSTRUCTIONS = """
Respond with a single valid JSON object adhering strictly to this schema:
{
  "page_summary": "string - Concise 1-2 sentence overview of the page content and current view",
  "visual_health_score": 0-100, // Integer score: 100 is pristine UI, <60 indicates severe visual flaws
  "issues": [
    {
      "issue_id": "string - e.g. 'ISSUE-001'",
      "category": "string - overlapping_text | misaligned_buttons | clipped_elements | responsive_layout | unreadable_text | missing_elements | contrast_issue | other",
      "severity": "string - critical | high | medium | low",
      "title": "string - short descriptive title of the defect",
      "location": "string - precise region on screen where the defect appears",
      "selector": "string - CSS selector of the culprit or target element (e.g. '.hero-content h1', '.cta-btn')",
      "description": "string - detailed explanation of what is visually broken",
      "suggested_fix": {
        "selector": "string - CSS selector",
        "css_properties": {
          "property-name": "value"
        },
        "css_code": "string - valid complete CSS rule (e.g. '.cta-btn { margin-top: 16px; position: relative; }')",
        "explanation": "string - concise technical explanation of how this fix resolves the defect"
      }
    }
  ],
  "css_fixes": {
    "selector_string": {
      "selector": "string - CSS selector",
      "properties": {
        "property-name": "value"
      },
      "css_rule": "string - complete CSS rule",
      "issue_id": "string - related issue_id",
      "description": "string - brief description of what this rule fixes"
    }
  },
  "ui_elements": {
    "buttons": [
      {
        "label": "string - visible button text or icon description",
        "type": "string - primary | secondary | icon_only | link_button | destructive",
        "location": "string - approximate screen location",
        "state": "string - normal | disabled | hovered | loading | unknown"
      }
    ],
    "input_fields": [
      {
        "placeholder": "string - placeholder text or label",
        "type": "string - text | password | email | search | select | textarea | checkbox",
        "location": "string - location on screen",
        "has_visible_label": true
      }
    ],
    "text_elements": [
      {
        "text_snippet": "string - first 5-8 words of text block",
        "role": "string - h1 | h2 | h3 | body | caption | badge | footer",
        "readability": "string - high | medium | low"
      }
    ],
    "navigation_elements": [
      {
        "label": "string - navigation link text or icon description",
        "type": "string - nav_link | breadcrumb | tab | dropdown | pagination",
        "location": "string - header | sidebar | footer"
      }
    ],
    "media_elements": [
      {
        "type": "string - image | icon | avatar | video_thumbnail | logo",
        "location": "string - location on screen",
        "status": "string - rendered_properly | placeholder | broken_render"
      }
    ]
  },
  "recommendations": [
    "string - general actionable improvements for UI polish, accessibility, and responsiveness"
  ]
}
"""


# ============================================================================
# Reusable Prompt Builders
# ============================================================================

def build_analysis_prompt(
    html_content: Optional[str] = None,
    page_url: Optional[str] = None,
    focus_areas: Optional[List[str]] = None,
    custom_notes: Optional[str] = None,
) -> str:
    """
    Constructs a visual analysis prompt for Google Gemini VLM.

    Args:
        html_content: Optional HTML snippet or full page DOM string to help correlate elements.
        page_url: Optional URL of the target page for context.
        focus_areas: Optional list of areas to inspect more intensely
                     (e.g., ['overlapping_text', 'misaligned_buttons', 'clipped_elements', 'responsive_layout']).
        custom_notes: Optional extra test notes.

    Returns:
        Formatted prompt string ready for Gemini VLM.
    """
    url_section = ""
    if page_url:
        url_section = f"\n### Target Page URL:\n{page_url.strip()}\n"

    html_section = ""
    if html_content:
        # Truncate extremely long HTML to protect token budgets (~25,000 chars)
        clean_html = html_content.strip()
        if len(clean_html) > 25000:
            clean_html = clean_html[:25000] + "\n<!-- [HTML content truncated for token limits] -->"
        html_section = f"\n### HTML DOM Context for Selector Correlation:\n```html\n{clean_html}\n```\n"

    focus_section = ""
    if focus_areas:
        focus_list = "\n".join(f"- {area}" for area in focus_areas)
        focus_section = f"\n### Priority Focus Areas for this Screenshot:\n{focus_list}\n"

    notes_section = ""
    if custom_notes:
        notes_section = f"\n### Additional Test Context:\n{custom_notes.strip()}\n"

    prompt = f"""{SYSTEM_PROMPT}
{url_section}{html_section}{focus_section}{notes_section}
### Visual Inspection Tasks:
1. Analyze the attached screenshot carefully for UI elements and visual bugs.
2. Specifically inspect for:
   - Overlapping text: Any text colliding with or obscuring other text, buttons, or cards.
   - Misaligned buttons: Buttons out of vertical/horizontal baseline alignment, uneven padding/margins.
   - Clipped elements: Overflowing text without ellipsis, cut-off buttons or cards, cropped icons.
   - Responsive layout problems: Broken columns, horizontal overflow, awkward wrapping on viewport boundaries.
3. For each defect found, generate a CSS fix dictionary with accurate CSS properties and a ready-to-use rule.
4. Populate the top-level 'css_fixes' dictionary mapping selectors to their structured rules.

### Output Schema:
{JSON_SCHEMA_INSTRUCTIONS}

Remember: Output ONLY valid JSON.
"""
    return prompt.strip()


# ============================================================================
# Mock VLM Response Generator (Zero API Key / Offline Testing)
# ============================================================================

def get_mock_vlm_response(
    screenshot: Optional[str] = None,
    html_content: Optional[str] = None,
    page_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates a realistic, fully-formed mock visual analysis result.
    Demonstrates overlapping text, misaligned buttons, clipped elements,
    and responsive layout problems with complete structured CSS fixes.
    """
    url = page_url or "https://app.omnisight.local/dashboard"
    screenshot_name = str(screenshot) if screenshot else "sample_screenshot.png"

    return {
        "page_summary": f"Web application dashboard for {url} showing navigation bar, hero analytics card, and action controls.",
        "visual_health_score": 72,
        "issues": [
            {
                "issue_id": "ISSUE-001",
                "category": "overlapping_text",
                "severity": "high",
                "title": "Hero subtitle overlaps primary action button",
                "location": "Hero banner, center-left",
                "selector": ".hero-subtitle",
                "description": "The subtitle text block line-height extends downwards and collides with the top 15px of the primary call-to-action button.",
                "suggested_fix": {
                    "selector": ".hero-subtitle",
                    "css_properties": {
                        "margin-bottom": "24px",
                        "line-height": "1.4"
                    },
                    "css_code": ".hero-subtitle { margin-bottom: 24px; line-height: 1.4; }",
                    "explanation": "Add bottom margin to push the button below the text block and normalize line height."
                }
            },
            {
                "issue_id": "ISSUE-002",
                "category": "misaligned_buttons",
                "severity": "medium",
                "title": "Secondary action button misaligned vertically with primary button",
                "location": "Hero banner action button group",
                "selector": ".btn-group .btn-secondary",
                "description": "Secondary 'Learn More' button has an inconsistent top margin offset of 6px relative to the 'Get Started' primary button.",
                "suggested_fix": {
                    "selector": ".btn-group",
                    "css_properties": {
                        "display": "inline-flex",
                        "align-items": "center",
                        "gap": "12px"
                    },
                    "css_code": ".btn-group { display: inline-flex; align-items: center; gap: 12px; }",
                    "explanation": "Align button group container with flexbox align-items center and a consistent 12px gap."
                }
            },
            {
                "issue_id": "ISSUE-003",
                "category": "clipped_elements",
                "severity": "high",
                "title": "Metric card title clipped by container boundary",
                "location": "Analytics summary grid, Card 2",
                "selector": ".metric-card-title",
                "description": "The title 'Total Converted Registrations (Q4)' is clipped horizontally because the parent container has overflow: hidden and fixed width.",
                "suggested_fix": {
                    "selector": ".metric-card-title",
                    "css_properties": {
                        "text-overflow": "ellipsis",
                        "white-space": "nowrap",
                        "overflow": "hidden",
                        "max-width": "100%"
                    },
                    "css_code": ".metric-card-title { text-overflow: ellipsis; white-space: nowrap; overflow: hidden; max-width: 100%; }",
                    "explanation": "Apply standard text truncation with ellipsis to prevent harsh clipping."
                }
            },
            {
                "issue_id": "ISSUE-004",
                "category": "responsive_layout",
                "severity": "medium",
                "title": "Horizontal overflow and broken column wrapping on smaller viewports",
                "location": "Metrics grid section",
                "selector": ".metrics-grid",
                "description": "The three-column metrics grid does not wrap, forcing a horizontal scrollbar on viewports under 1024px.",
                "suggested_fix": {
                    "selector": ".metrics-grid",
                    "css_properties": {
                        "display": "grid",
                        "grid-template-columns": "repeat(auto-fit, minmax(280px, 1fr))",
                        "gap": "16px"
                    },
                    "css_code": ".metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }",
                    "explanation": "Use responsive auto-fit grid columns with a 280px minimum to automatically wrap without horizontal overflow."
                }
            }
        ],
        "css_fixes": {
            ".hero-subtitle": {
                "selector": ".hero-subtitle",
                "properties": {
                    "margin-bottom": "24px",
                    "line-height": "1.4"
                },
                "css_rule": ".hero-subtitle { margin-bottom: 24px; line-height: 1.4; }",
                "issue_id": "ISSUE-001",
                "description": "Add bottom margin to prevent text from overlapping primary CTA button"
            },
            ".btn-group": {
                "selector": ".btn-group",
                "properties": {
                    "display": "inline-flex",
                    "align-items": "center",
                    "gap": "12px"
                },
                "css_rule": ".btn-group { display: inline-flex; align-items: center; gap: 12px; }",
                "issue_id": "ISSUE-002",
                "description": "Align button group container with flexbox align-items center"
            },
            ".metric-card-title": {
                "selector": ".metric-card-title",
                "properties": {
                    "text-overflow": "ellipsis",
                    "white-space": "nowrap",
                    "overflow": "hidden",
                    "max-width": "100%"
                },
                "css_rule": ".metric-card-title { text-overflow: ellipsis; white-space: nowrap; overflow: hidden; max-width: 100%; }",
                "issue_id": "ISSUE-003",
                "description": "Apply standard text truncation with ellipsis to prevent harsh clipping"
            },
            ".metrics-grid": {
                "selector": ".metrics-grid",
                "properties": {
                    "display": "grid",
                    "grid-template-columns": "repeat(auto-fit, minmax(280px, 1fr))",
                    "gap": "16px"
                },
                "css_rule": ".metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }",
                "issue_id": "ISSUE-004",
                "description": "Use responsive auto-fit grid columns with 280px minimum"
            }
        },
        "ui_elements": {
            "buttons": [
                {
                    "label": "Get Started",
                    "type": "primary",
                    "location": "hero section left",
                    "state": "normal"
                },
                {
                    "label": "Learn More",
                    "type": "secondary",
                    "location": "hero section next to CTA",
                    "state": "normal"
                },
                {
                    "label": "Export Report",
                    "type": "icon_only",
                    "location": "top-right header",
                    "state": "normal"
                }
            ],
            "input_fields": [
                {
                    "placeholder": "Filter analytics by date or keyword...",
                    "type": "search",
                    "location": "dashboard header center",
                    "has_visible_label": False
                }
            ],
            "text_elements": [
                {
                    "text_snippet": "OmniSight Performance Dashboard",
                    "role": "h1",
                    "readability": "high"
                },
                {
                    "text_snippet": "Real-time automated visual regression and QA inspection metrics.",
                    "role": "body",
                    "readability": "medium"
                }
            ],
            "navigation_elements": [
                {
                    "label": "Overview",
                    "type": "nav_link",
                    "location": "header"
                },
                {
                    "label": "Test Runs",
                    "type": "nav_link",
                    "location": "header"
                },
                {
                    "label": "Settings",
                    "type": "nav_link",
                    "location": "header"
                }
            ],
            "media_elements": [
                {
                    "type": "logo",
                    "location": "top-left header",
                    "status": "rendered_properly"
                },
                {
                    "type": "image",
                    "location": "analytics chart container",
                    "status": "rendered_properly"
                }
            ]
        },
        "recommendations": [
            "Add margin-bottom to .hero-subtitle to prevent overlapping CTA buttons.",
            "Use flexbox alignment on .btn-group to keep sibling action buttons vertically centered.",
            "Add text-overflow: ellipsis to .metric-card-title to prevent container overflow clipping.",
            "Use CSS Grid auto-fit on .metrics-grid to support responsive wrapping on tablet and mobile viewports."
        ],
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "screenshot_path": screenshot_name,
            "page_url": url,
            "model": "mock-vlm-engine",
            "status": "success",
            "mock_mode": True,
            "duration_seconds": 0.05
        }
    }
