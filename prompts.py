"""
OmniSight - Member 2: AI/VLM Basic Setup and Prompt Testing
Prompts Module (prompts.py)

Contains reusable system prompts, inspection prompt builders, and JSON schema definitions
for analyzing website screenshots with Vision-Language Models (VLMs).
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone


# ============================================================================
# System Instructions
# ============================================================================

SYSTEM_PROMPT = """You are OmniSight's Senior Automated QA & UI/UX Accessibility Vision Inspector.
Your mission is to meticulously analyze website screenshots captured during automated browser testing.

You evaluate screenshots across two primary pillars:
1. UI Element Inventory: Detect buttons, text elements, input fields, navigation controls, and media.
2. Visual & Structural Defect Detection: Identify visual bugs such as:
   - Overlapping elements (e.g., text overlapping buttons, badges overlapping headers).
   - Unreadable text (e.g., poor contrast ratio, text truncated by container, microscopic font).
   - Missing visible elements (e.g., broken image placeholders, empty containers, missing icons).
   - Layout & responsiveness issues (e.g., misaligned columns, awkward whitespace, viewport overflow).

You MUST return your findings in strict, valid JSON matching the exact schema specified.
Do NOT include markdown formatting, conversational commentary, or text outside the JSON object.
"""


# ============================================================================
# Output JSON Schema Specification
# ============================================================================

JSON_SCHEMA_INSTRUCTIONS = """
Respond with a single valid JSON object adhering strictly to this schema:
{
  "page_summary": "string - Concise 1-2 sentence overview of the page content and current view",
  "visual_health_score": 0-100, // Integer score: 100 is pristine UI, <60 indicates severe visual flaws
  "ui_elements": {
    "buttons": [
      {
        "label": "string - visible button text or icon description",
        "type": "string - primary | secondary | icon_only | link_button | destructive",
        "location": "string - approximate screen location (e.g., 'top-right header', 'hero section')",
        "state": "string - normal | disabled | hovered | loading | unknown"
      }
    ],
    "input_fields": [
      {
        "placeholder": "string - placeholder text or label",
        "type": "string - text | password | email | search | select | textarea | checkbox",
        "location": "string - location on screen",
        "has_visible_label": true | false
      }
    ],
    "text_elements": [
      {
        "text_snippet": "string - first 5-8 words of significant text block",
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
  "visual_issues": [
    {
      "issue_id": "string - e.g. 'ISSUE-001'",
      "category": "string - overlapping_elements | unreadable_text | missing_elements | layout_issue | contrast_issue | other",
      "severity": "string - critical | high | medium | low",
      "title": "string - short descriptive title of the defect",
      "location": "string - precise region where the issue appears",
      "description": "string - detailed explanation of what is visually broken",
      "recommendation": "string - suggested CSS or layout fix"
    }
  ],
  "recommendations": [
    "string - general actionable improvements for UI polish, accessibility, and responsiveness"
  ]
}
"""


# ============================================================================
# Reusable Prompt Builders
# ============================================================================

def build_analysis_prompt(
    focus_areas: Optional[List[str]] = None,
    custom_notes: Optional[str] = None
) -> str:
    """
    Constructs a comprehensive visual analysis prompt for the VLM.

    Args:
        focus_areas: Optional list of areas to inspect more intensely
                     (e.g., ['overlapping_elements', 'mobile_responsiveness', 'buttons']).
        custom_notes: Optional extra context from test runner or user.

    Returns:
        Formatted prompt string ready for VLM consumption.
    """
    focus_section = ""
    if focus_areas:
        focus_list = "\n".join(f"- {area}" for area in focus_areas)
        focus_section = f"\n### High-Priority Focus Areas for this Screenshot:\n{focus_list}\n"

    notes_section = ""
    if custom_notes:
        notes_section = f"\n### Additional Test Context:\n{custom_notes}\n"

    prompt = f"""{SYSTEM_PROMPT}

### Analysis Tasks:
1. Identify all interactive and structural UI elements:
   - Buttons (labels, types, states, locations)
   - Input fields (placeholders, types, locations, label presence)
   - Key text blocks (headings, body text, readability)
   - Navigation links and menus
   - Media (images, logos, icons, avatars)

2. Inspect thoroughly for visual and layout defects:
   - Overlapping elements (text collisions, card overlapping, banner obscuring elements)
   - Unreadable text (low contrast against background, clipped/truncated text, tiny fonts)
   - Missing or broken visible elements (empty containers, broken images, unrendered icons)
   - Layout issues (misalignments, horizontal viewport overflow, awkward whitespace)
{focus_section}{notes_section}
### Output Schema:
{JSON_SCHEMA_INSTRUCTIONS}

Remember: Output ONLY valid JSON.
"""
    return prompt.strip()


# ============================================================================
# Mock Analysis Data Generator (for Offline Testing and Unit Tests)
# ============================================================================

def get_mock_analysis_data(screenshot_path: str = "sample_screenshot.png") -> Dict[str, Any]:
    """
    Generates a realistic, fully-formed mock analysis result.
    Useful for unit testing without API keys, CI/CD pipelines, and early member integration.
    """
    return {
        "page_summary": "E-commerce landing page featuring a promotional hero section, navigation bar, and product grid.",
        "visual_health_score": 78,
        "ui_elements": {
            "buttons": [
                {
                    "label": "Shop Now",
                    "type": "primary",
                    "location": "hero section center",
                    "state": "normal"
                },
                {
                    "label": "Sign In",
                    "type": "secondary",
                    "location": "top-right header",
                    "state": "normal"
                },
                {
                    "label": "Add to Cart",
                    "type": "primary",
                    "location": "product card 1",
                    "state": "normal"
                }
            ],
            "input_fields": [
                {
                    "placeholder": "Search products, brands and categories...",
                    "type": "search",
                    "location": "header center",
                    "has_visible_label": False
                },
                {
                    "placeholder": "Enter your email for 10% off",
                    "type": "email",
                    "location": "newsletter footer",
                    "has_visible_label": True
                }
            ],
            "text_elements": [
                {
                    "text_snippet": "Summer Collection 2026 - Discover New Arrivals",
                    "role": "h1",
                    "readability": "high"
                },
                {
                    "text_snippet": "Limited time offer. Free shipping over $50.",
                    "role": "body",
                    "readability": "medium"
                },
                {
                    "text_snippet": "Featured Best Sellers",
                    "role": "h2",
                    "readability": "high"
                }
            ],
            "navigation_elements": [
                {
                    "label": "Home",
                    "type": "nav_link",
                    "location": "header"
                },
                {
                    "label": "Categories",
                    "type": "dropdown",
                    "location": "header"
                },
                {
                    "label": "About Us",
                    "type": "nav_link",
                    "location": "footer"
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
                    "location": "hero banner background",
                    "status": "rendered_properly"
                },
                {
                    "type": "image",
                    "location": "product card 3",
                    "status": "placeholder"
                }
            ]
        },
        "visual_issues": [
            {
                "issue_id": "ISSUE-001",
                "category": "overlapping_elements",
                "severity": "high",
                "title": "Hero subtitle overlaps primary CTA button",
                "location": "Hero section, center right",
                "description": "The description paragraph extends downwards and overlaps the top 20% of the 'Shop Now' button on smaller viewports.",
                "recommendation": "Add `margin-bottom: 1.5rem` to the subtitle paragraph or wrap in a flex column layout with consistent gap."
            },
            {
                "issue_id": "ISSUE-002",
                "category": "unreadable_text",
                "severity": "medium",
                "title": "Low contrast on footer newsletter disclaimer",
                "location": "Footer section, bottom center",
                "description": "Light grey text (#A0A0A0) on white background fails WCAG 2.1 AA minimum contrast requirement (2.4:1 ratio).",
                "recommendation": "Darken font color to at least #595959 to achieve minimum 4.5:1 contrast ratio."
            },
            {
                "issue_id": "ISSUE-003",
                "category": "missing_elements",
                "severity": "medium",
                "title": "Product card 3 displays broken/empty image placeholder",
                "location": "Product grid, item 3",
                "description": "Image element has an unrendered gray placeholder box instead of a product image.",
                "recommendation": "Verify image asset URL and implement a styled fallback icon for missing product media."
            }
        ],
        "recommendations": [
            "Fix hero section flexbox gap to prevent button overlap across different viewport widths.",
            "Increase text contrast in footer disclaimer to meet WCAG AA standards.",
            "Verify product image asset pipeline to ensure all grid items load high-resolution images."
        ]
    }
