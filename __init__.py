"""
OmniSight - Member 2: AI Integration using Vision-Language Models (VLM)
AI Package (ai)

Provides automated visual QA inspection for website screenshots using Google Gemini VLM.
Detects visual defects including overlapping text, misaligned buttons, clipped elements,
and responsive layout problems, returning structured issue data and CSS fixes.

Week 4 VLM Optimization additions:
- ImageChunker: Slices large/tall screenshots into overlapping chunks with blank detection
- ImageCropper: Crops to anomalous/suspicious UI components
- VLMTokenOptimizer: Optimizes payload and calculates token reductions
- PerformanceTracker: Measures latency benchmarks and speedups
- VLMOptimizer: Unified optimization engine
"""

from ai.vision_analyzer import (
    VisionAnalyzer,
    analyze_screenshot,
    verify_fix,
    analyze_optimized,
)
from ai.prompts import (
    SYSTEM_PROMPT,
    JSON_SCHEMA_INSTRUCTIONS,
    build_analysis_prompt,
    get_mock_vlm_response,
    build_verification_prompt,
    get_mock_verification_response,
    build_roi_analysis_prompt,
    build_chunk_analysis_prompt,
    get_mock_roi_response,
)
from ai.optimizer import (
    VLMOptimizer,
    ImageChunker,
    ImageCropper,
    VLMTokenOptimizer,
    PerformanceTracker,
    ChunkMetadata,
    ROICandidate,
    TokenMetrics,
    PerformanceMetrics,
)

__all__ = [
    "VisionAnalyzer",
    "analyze_screenshot",
    "verify_fix",
    "analyze_optimized",
    "SYSTEM_PROMPT",
    "JSON_SCHEMA_INSTRUCTIONS",
    "build_analysis_prompt",
    "get_mock_vlm_response",
    "build_verification_prompt",
    "get_mock_verification_response",
    "build_roi_analysis_prompt",
    "build_chunk_analysis_prompt",
    "get_mock_roi_response",
    "VLMOptimizer",
    "ImageChunker",
    "ImageCropper",
    "VLMTokenOptimizer",
    "PerformanceTracker",
    "ChunkMetadata",
    "ROICandidate",
    "TokenMetrics",
    "PerformanceMetrics",
]
