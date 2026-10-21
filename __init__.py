"""
OmniSight - Member 2: AI Integration using Vision-Language Models (VLM)
AI Package (ai)

Provides automated visual QA inspection for website screenshots using Google Gemini VLM.
Detects visual defects including overlapping text, misaligned buttons, clipped elements,
and responsive layout problems, returning structured issue data and CSS fixes.
"""

from ai.vision_analyzer import VisionAnalyzer, analyze_screenshot
from ai.prompts import (
    SYSTEM_PROMPT,
    JSON_SCHEMA_INSTRUCTIONS,
    build_analysis_prompt,
    get_mock_vlm_response,
)

__all__ = [
    "VisionAnalyzer",
    "analyze_screenshot",
    "SYSTEM_PROMPT",
    "JSON_SCHEMA_INSTRUCTIONS",
    "build_analysis_prompt",
    "get_mock_vlm_response",
]
