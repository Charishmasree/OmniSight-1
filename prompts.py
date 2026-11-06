"""
OmniSight - Member 2: AI Integration using Vision-Language Models (VLM)
Prompts Module (ai/prompts.py)

Contains reusable system prompts, inspection prompt builders, and JSON schema definitions
for analyzing website screenshots with Google Gemini VLM.
Instructs the model to detect visual UI issues such as overlapping text, misaligned buttons,
clipped elements, and responsive layout problems, and to generate structured CSS fixes.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone


# ============================================================================
# System Instructions
# ============================================================================

SYSTEM_PROMPT = """You are OmniSight's Senior Automated QA & UI/UX Accessibility Vision Inspector.
Your mission is to meticulously analyze website screenshots captured during automated browser testing,
optionally assisted by HTML DOM context and page URL.

You evaluate the screenshot across two primary pillars:
1. UI Element Inventory: Detect buttons, text elements, input fields, navigation controls, and media.
2. Visual & Structural Defect Detection: Specifically diagnose UI defects including:
   - overlapping_text: Text colliding with, obscuring, or placed awkwardly over other text, images, or buttons.
   - misaligned_buttons: Buttons out of alignment with grid, flex containers, text baselines, or sibling buttons.
   - clipped_elements: Text truncated by overflow without ellipsis, cut-off containers, or clipped badges/icons.
   - responsive_layout: Horizontal scrollbar caused by element overflow, awkward line wrapping, broken columns, or viewport scaling bugs.
   - unreadable_text: Low contrast ratio against background, microscopic font size, or unreadable styling.
   - missing_elements: Broken image icons, empty containers, missing SVGs, or unrendered placeholders.

For every visual defect detected, you MUST provide:
- A specific CSS selector for the affected element (inferred from provided HTML or standard semantic naming).
- A structured suggested CSS fix containing specific CSS property-value pairs and a valid CSS rule.
- A concise explanation of why the CSS fix resolves the defect.

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


# ============================================================================
# Self-Healing Verification Prompts (Week 3 Continuity)
# ============================================================================

def build_verification_prompt(
    original_issue: Dict[str, Any],
    applied_css: str,
    attempt: int = 1,
    max_attempts: int = 3,
    html_content: Optional[str] = None,
) -> str:
    """
    Constructs a visual verification prompt for Google Gemini VLM to evaluate
    whether an applied CSS fix resolved a specific UI defect.
    """
    selector = original_issue.get("selector", ".target-element")
    category = original_issue.get("category", "ui_defect")
    title = original_issue.get("title", original_issue.get("description", "UI visual defect"))
    desc = original_issue.get("description", "")

    html_section = ""
    if html_content:
        clean_html = html_content.strip()
        if len(clean_html) > 15000:
            clean_html = clean_html[:15000] + "\n<!-- [HTML truncated] -->"
        html_section = f"\n### Active HTML Context:\n```html\n{clean_html}\n```\n"

    prompt = f"""You are OmniSight's Autonomous Self-Healing Verification Vision Inspector.
Your objective is to inspect the attached screenshot captured AFTER a candidate CSS fix was applied.
Determine whether the reported visual defect has been fixed, partially fixed, or remains unresolved.

### Target Defect Details:
- Issue Title: {title}
- Defect Category: {category}
- Culprit Selector: {selector}
- Original Description: {desc}
- Attempt Number: {attempt} of {max_attempts}

### Applied CSS Fix:
```css
{applied_css}
```
{html_section}
### Evaluation Instructions:
1. Carefully inspect the updated screenshot at the location of '{selector}'.
2. Classify verification_status as EXACTLY one of:
   - "fixed": The defect is completely resolved. No collision, clipping, or visual flaw remains.
   - "partially_fixed": Improvement is visible, but secondary visual defects remain.
   - "unresolved": The defect is still clearly present.
3. If not fully fixed and attempt < {max_attempts}, propose a revised CSS fix.
4. Output strict JSON matching this schema:
{{
  "verification_status": "fixed" | "partially_fixed" | "unresolved",
  "verification_reason": "string - precise technical explanation of visual state",
  "defect_resolved": true | false,
  "visual_health_score": 0-100,
  "remaining_issues": [
    {{"description": "string", "severity": "high" | "medium" | "low"}}
  ],
  "revised_css_suggestion": "string or null"
}}

Respond ONLY with valid JSON.
"""
    return prompt.strip()


def get_mock_verification_response(
    original_issue: Dict[str, Any],
    applied_css: str,
    attempt: int = 1,
    max_attempts: int = 3,
    force_status: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns realistic synthetic verification output for offline testing without an API key.
    """
    selector = original_issue.get("selector", ".target-element")
    title = original_issue.get("title", original_issue.get("description", "UI defect"))

    status = (force_status or ("fixed" if attempt <= max_attempts else "unresolved")).lower()

    if status == "fixed":
        reason = f"Defect resolved. Element '{selector}' is properly displayed with no overlapping text, clipping, or misalignment in the updated screenshot."
        defect_resolved = True
        health_score = 92
        remaining_issues = []
        revised_css = None
    elif status == "partially_fixed":
        reason = f"Element '{selector}' partially resolved, but secondary alignment offset remains."
        defect_resolved = False
        health_score = 75
        remaining_issues = [{"description": "Minor residual alignment offset", "severity": "low"}]
        revised_css = f"{selector} {{ margin-bottom: 28px !important; line-height: 1.5 !important; }}"
    else:
        status = "unresolved"
        reason = f"Defect persists. Element '{selector}' still exhibits visual defects after applying the suggested CSS."
        defect_resolved = False
        health_score = 65
        remaining_issues = [{"description": f"{title} remains unresolved.", "severity": "high"}]
        revised_css = f"{selector} {{ margin-bottom: 32px !important; line-height: 1.6 !important; }}"

    return {
        "verification_status": status,
        "verification_reason": reason,
        "defect_resolved": defect_resolved,
        "visual_health_score": health_score,
        "remaining_issues": remaining_issues,
        "revised_css_suggestion": revised_css,
    }


# ============================================================================
# Week 4 VLM Optimization Prompts (Chunking, Cropping, and Token Reduction)
# ============================================================================

def build_roi_analysis_prompt(
    roi_label: str,
    roi_bbox: Tuple[int, int, int, int],
    html_snippet: Optional[str] = None,
    expected_defect_types: Optional[List[str]] = None,
) -> str:
    """
    Constructs a highly focused analysis prompt for a cropped region of interest (ROI).
    Optimized for token reduction and low latency by restricting context to the crop bounds.

    Args:
        roi_label: Description of the region (e.g. 'hero_cta_badge', 'card_title').
        roi_bbox: Bounding box tuple (left, top, right, bottom).
        html_snippet: Optional relevant snippet of HTML for the cropped element.
        expected_defect_types: Optional list of suspected defect categories.

    Returns:
        Formatted prompt string tailored for micro-region inspection.
    """
    defects = ", ".join(expected_defect_types or [
        "overlapping_text", "clipped_elements", "misaligned_buttons", "hidden_elements", "responsive_layout"
    ])
    html_section = f"\n### Local DOM Subtree:\n```html\n{html_snippet.strip()}\n```\n" if html_snippet else ""

    return f"""You are OmniSight's Micro-Region Vision QA Specialist (VLM Optimization Mode).
Analyze this CROPPED screenshot region focusing strictly on:
- Bounding Box: {roi_bbox}
- Region Label: '{roi_label}'
- Potential Defect Categories: {defects}
{html_section}
Perform a micro-region inspection to determine if an anomaly exists inside this crop:
1. Overlapping text: Collision between text spans, buttons, badges, or icons.
2. Clipped elements: Overflowing text without ellipsis or clipped container edges.
3. Misaligned buttons: Uneven padding, baseline offsets, or misaligned button groups.
4. Hidden UI elements: Collapsed containers or zero-dimension content.
5. Responsive layout bugs: Wrapping collisions or broken horizontal flow.

Output strict JSON with:
{{
  "region_label": "{roi_label}",
  "bbox": list({roi_bbox}),
  "defect_detected": true | false,
  "category": "overlapping_text | misaligned_buttons | clipped_elements | hidden_elements | responsive_layout | none",
  "severity": "critical | high | medium | low | none",
  "title": "string",
  "selector": "string - precise CSS selector",
  "description": "string - detailed defect description",
  "suggested_fix": {{
    "selector": "string",
    "css_properties": {{"prop": "value"}},
    "css_code": "string",
    "explanation": "string"
  }},
  "visual_health_score": 0-100
}}

Respond ONLY with valid JSON.
""".strip()


def build_chunk_analysis_prompt(
    chunk_index: int,
    total_chunks: int,
    bbox: Tuple[int, int, int, int],
    page_url: Optional[str] = None,
) -> str:
    """
    Constructs an optimized analysis prompt for an individual screenshot chunk.
    Coordinates allow correlating detected defects back to the global viewport.

    Args:
        chunk_index: Index of current chunk (0-based).
        total_chunks: Total number of chunks generated.
        bbox: Absolute bounding box (left, top, right, bottom) of this chunk.
        page_url: Optional target page URL.

    Returns:
        Formatted chunk inspection prompt.
    """
    url_line = f"Page URL: {page_url}\n" if page_url else ""
    return f"""You are OmniSight's Chunk-Level Vision Inspector.
{url_line}Currently inspecting Chunk {chunk_index + 1} of {total_chunks}.
Chunk Global Coordinates: Bounding Box (left={bbox[0]}, top={bbox[1]}, right={bbox[2]}, bottom={bbox[3]}).

Analyze this specific slice for UI defects:
- Overlapping text
- Clipped buttons or card containers
- Misaligned interactive elements
- Responsive layout defects

Output strict JSON adhering to the standard OmniSight issue schema.
""".strip()


def get_mock_roi_response(
    roi_label: str,
    bbox: Tuple[int, int, int, int],
    defect_category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns realistic synthetic QA analysis for a cropped region without calling external API.
    """
    category = defect_category or "overlapping_text"
    selector = f".{roi_label.replace('_', '-')}"
    return {
        "region_label": roi_label,
        "bbox": list(bbox),
        "defect_detected": True,
        "category": category,
        "severity": "high",
        "title": f"Visual defect in {roi_label}",
        "selector": selector,
        "description": f"Visual anomaly ({category}) detected inside cropped region {bbox}.",
        "suggested_fix": {
            "selector": selector,
            "css_properties": {"margin-bottom": "24px", "line-height": "1.4"},
            "css_code": f"{selector} {{ margin-bottom: 24px; line-height: 1.4; }}",
            "explanation": f"Normalizes spacing and layout for {roi_label}.",
        },
        "visual_health_score": 75,
    }

