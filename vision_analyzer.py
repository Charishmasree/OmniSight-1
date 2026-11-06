"""
OmniSight - Member 2: AI Integration using Vision-Language Models (VLM)
AI Vision Analyzer (ai/vision_analyzer.py)

Core engine for analyzing website screenshots using Google Gemini Vision-Language Models (VLM).
Detects UI defects such as overlapping text, misaligned buttons, clipped elements,
and responsive layout problems, returning structured findings with ready-to-apply CSS fixes.
"""

import io
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

from config import (
    VisionConfig,
    get_config,
    OmniSightVisionError,
    APIKeyMissingError,
    ImageNotFoundError,
    InvalidImageFormatError,
    ImageCorruptedError,
    ImageSizeExceededError,
    ModelInferenceError,
    ResponseParseError,
)
from ai.prompts import (
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

# Load environment variables from .env if present
load_dotenv()

# Configure module logger
logger = logging.getLogger("omnisight.ai.vision_analyzer")


class VisionAnalyzer:
    """
    Vision-Language Model (VLM) analyzer for automated website screenshot QA.

    Uses Google Gemini VLM to inspect screenshots for visual defects:
    - Overlapping text
    - Misaligned buttons
    - Clipped elements
    - Responsive layout problems

    Generates structured issue lists and CSS fixes for Action Engine (Member 3)
    and FastAPI backend (Member 4).
    """

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    MIME_TYPES = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        mock_mode: Optional[bool] = None,
        config: Optional[VisionConfig] = None,
    ):
        """
        Initialize the VisionAnalyzer.

        Args:
            api_key: Optional Gemini API key. Defaults to GEMINI_API_KEY environment variable.
                     Never hardcodes secrets.
            model_name: Model identifier (e.g. 'gemini-2.5-flash', 'gemini-2.0-flash').
            mock_mode: When True, bypasses API calls and returns realistic synthetic QA findings.
            config: Optional pre-configured VisionConfig instance.
        """
        overrides = {}
        # Read API key strictly from environment if not provided explicitly
        resolved_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or ""
        )
        if resolved_key:
            overrides["gemini_api_key"] = resolved_key

        if model_name is not None:
            overrides["model_name"] = model_name

        # Detect mock mode from argument or environment variable
        if mock_mode is not None:
            overrides["mock_mode"] = mock_mode
        elif os.getenv("VLM_MOCK_MODE", "").lower() in ("true", "1", "yes"):
            overrides["mock_mode"] = True

        self.config = config or get_config(**overrides)
        # Ensure gemini_api_key matches resolved key if passed
        if resolved_key and not self.config.gemini_api_key:
            self.config.gemini_api_key = resolved_key

        self._client = None
        self.optimizer = VLMOptimizer()

        if not self.config.mock_mode:
            self.config.validate()
            self._init_client()
        else:
            logger.info("VisionAnalyzer initialized in MOCK MODE (no external API calls).")

    def _init_client(self) -> None:
        """Initializes the google.genai Client."""
        try:
            from google import genai
            self._client = genai.Client(api_key=self.config.gemini_api_key)
            logger.info(f"Initialized Google GenAI Client with model: {self.config.model_name}")
        except Exception as e:
            raise ModelInferenceError(f"Failed to initialize Google GenAI Client: {e}") from e

    def _prepare_image_data(
        self,
        screenshot: Union[str, Path, bytes, Image.Image],
    ) -> Tuple[bytes, str, Dict[str, Any], Optional[str]]:
        """
        Validates and processes screenshot input from multiple types:
        - str or Path (filesystem path)
        - bytes (raw image bytes)
        - PIL Image

        Returns:
            Tuple of (image_bytes, mime_type, image_info_dict, filename_or_label)
        """
        max_bytes = int(self.config.max_image_size_mb * 1024 * 1024)

        # Case 1: Filesystem path (str or Path)
        if isinstance(screenshot, (str, Path)):
            path = Path(screenshot).resolve()
            if not path.exists():
                raise ImageNotFoundError(f"Screenshot file does not exist: {path}")
            if not path.is_file():
                raise ImageNotFoundError(f"Screenshot path is not a file: {path}")

            ext = path.suffix.lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                raise InvalidImageFormatError(
                    f"Unsupported image extension '{ext}'. Supported: {sorted(list(self.SUPPORTED_EXTENSIONS))}"
                )

            file_size = path.stat().st_size
            if file_size > max_bytes:
                size_mb = file_size / (1024 * 1024)
                raise ImageSizeExceededError(
                    f"Image size ({size_mb:.2f} MB) exceeds maximum allowed {self.config.max_image_size_mb} MB."
                )

            try:
                with Image.open(path) as img:
                    img.verify()
                with Image.open(path) as img:
                    width, height = img.size
                    img_format = img.format or ext.lstrip(".").upper()
            except (UnidentifiedImageError, OSError, Exception) as e:
                raise ImageCorruptedError(f"Corrupted or invalid image file: {path}. Details: {e}") from e

            image_bytes = path.read_bytes()
            mime_type = self.MIME_TYPES.get(ext, "image/png")
            info = {
                "width": width,
                "height": height,
                "format": img_format,
                "file_size_bytes": file_size,
            }
            return image_bytes, mime_type, info, path.name

        # Case 2: Raw bytes
        elif isinstance(screenshot, (bytes, bytearray)):
            image_bytes = bytes(screenshot)
            if len(image_bytes) == 0:
                raise ImageCorruptedError("Screenshot byte buffer is empty (0 bytes).")
            if len(image_bytes) > max_bytes:
                size_mb = len(image_bytes) / (1024 * 1024)
                raise ImageSizeExceededError(
                    f"Image bytes size ({size_mb:.2f} MB) exceeds limit {self.config.max_image_size_mb} MB."
                )

            try:
                bio = io.BytesIO(image_bytes)
                with Image.open(bio) as img:
                    img.verify()
                bio.seek(0)
                with Image.open(bio) as img:
                    width, height = img.size
                    img_format = img.format or "PNG"
            except (UnidentifiedImageError, OSError, Exception) as e:
                raise ImageCorruptedError(f"Corrupted or invalid screenshot bytes: {e}") from e

            mime_type = "image/png" if img_format == "PNG" else "image/jpeg"
            info = {
                "width": width,
                "height": height,
                "format": img_format,
                "file_size_bytes": len(image_bytes),
            }
            return image_bytes, mime_type, info, "in_memory_screenshot.png"

        # Case 3: PIL Image
        elif isinstance(screenshot, Image.Image):
            width, height = screenshot.size
            img_format = screenshot.format or "PNG"
            bio = io.BytesIO()
            screenshot.save(bio, format=img_format if img_format in ("PNG", "JPEG") else "PNG")
            image_bytes = bio.getvalue()
            mime_type = "image/png" if img_format == "PNG" else "image/jpeg"
            info = {
                "width": width,
                "height": height,
                "format": img_format,
                "file_size_bytes": len(image_bytes),
            }
            return image_bytes, mime_type, info, "pil_image.png"

        else:
            raise InvalidImageFormatError(
                f"Invalid screenshot parameter type '{type(screenshot).__name__}'. "
                "Expected str, Path, bytes, or PIL Image."
            )

    def _clean_and_parse_json(self, raw_text: str) -> Dict[str, Any]:
        """
        Strips markdown code fences and parses raw model output as JSON.

        Args:
            raw_text: Raw string returned by the VLM.

        Returns:
            Parsed Python dictionary.

        Raises:
            ResponseParseError: If parsing fails or output is not a JSON object.
        """
        cleaned = raw_text.strip()

        # Remove markdown code block fences (e.g. ```json ... ```)
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from VLM response: {raw_text[:200]}...")
            raise ResponseParseError(
                f"Model response was not valid JSON: {e}. Raw content preview: {cleaned[:150]}"
            ) from e

        if not isinstance(data, dict):
            raise ResponseParseError(f"Expected JSON object (dict), but received: {type(data).__name__}")

        # Standardize issues key across both 'issues' and 'visual_issues'
        if "issues" not in data and "visual_issues" in data:
            data["issues"] = data["visual_issues"]
        elif "visual_issues" not in data and "issues" in data:
            data["visual_issues"] = data["issues"]

        data.setdefault("page_summary", "Page visual analysis completed.")
        data.setdefault("visual_health_score", 80)
        data.setdefault("issues", [])
        data.setdefault("css_fixes", {})
        data.setdefault("ui_elements", {})
        data.setdefault("recommendations", [])

        # Ensure structured css_fixes dictionary is populated from issues if empty
        if not data["css_fixes"] and data.get("issues"):
            fixes = {}
            for issue in data["issues"]:
                suggested = issue.get("suggested_fix")
                if isinstance(suggested, dict):
                    sel = suggested.get("selector") or issue.get("selector") or issue.get("issue_id", "general")
                    fixes[sel] = {
                        "selector": sel,
                        "properties": suggested.get("css_properties", {}),
                        "css_rule": suggested.get("css_code", ""),
                        "issue_id": issue.get("issue_id", ""),
                        "description": suggested.get("explanation") or issue.get("title", ""),
                    }
            data["css_fixes"] = fixes

        return data

    def _call_gemini_api(self, image_bytes: bytes, mime_type: str, prompt_text: str) -> str:
        """
        Sends the screenshot bytes and prompt to Google Gemini via google.genai SDK.

        Args:
            image_bytes: Raw screenshot bytes.
            mime_type: MIME type string.
            prompt_text: Formatted prompt instructions.

        Returns:
            Raw response text from model.

        Raises:
            ModelInferenceError: If network, auth, or model call fails.
        """
        from google.genai import types

        try:
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=self.config.temperature,
            )

            response = self._client.models.generate_content(
                model=self.config.model_name,
                contents=[image_part, prompt_text],
                config=config,
            )

            if not response or not response.text:
                raise ModelInferenceError("VLM returned an empty or null response.")

            return response.text

        except Exception as e:
            logger.error(f"Error during Gemini VLM inference: {e}")
            raise ModelInferenceError(f"Gemini VLM inference failed: {e}") from e

    def analyze_screenshot(
        self,
        screenshot: Union[str, Path, bytes, Image.Image],
        html_content: Optional[str] = None,
        page_url: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
        custom_notes: Optional[str] = None,
        custom_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes a website screenshot using Google Gemini VLM.

        Detects UI defects such as:
        - Overlapping text
        - Misaligned buttons
        - Clipped elements
        - Responsive layout problems

        Generates structured suggested CSS fixes.

        Args:
            screenshot: Path to screenshot file, raw bytes, or PIL Image.
            html_content: Optional HTML DOM string to aid selector and element correlation.
            page_url: Optional URL of the target page for context.
            focus_areas: Optional priority areas (e.g. ['overlapping_text', 'misaligned_buttons']).
            custom_notes: Optional extra notes for QA context.
            custom_prompt: Optional complete prompt override.

        Returns:
            Structured dictionary conforming to the OmniSight VLM QA schema with:
            - status: "success"
            - page_url: str or None
            - page_summary: str
            - visual_health_score: int (0-100)
            - issues: list of detected visual defects with structured CSS fixes
            - css_fixes: structured dictionary mapping selectors to CSS rules and properties
            - ui_elements: detected buttons, input fields, text, navigation, media
            - recommendations: list of actionable UI polish steps
            - metadata: execution timestamp, model info, dimensions, duration

        Raises:
            ImageNotFoundError, InvalidImageFormatError, ImageCorruptedError,
            ImageSizeExceededError, ModelInferenceError, ResponseParseError, APIKeyMissingError.
        """
        start_time = time.time()

        # Step 1: Validate screenshot and get image info
        image_bytes, mime_type, image_info, image_name = self._prepare_image_data(screenshot)

        # Step 2: Build prompt with HTML and URL context
        prompt = custom_prompt or build_analysis_prompt(
            html_content=html_content,
            page_url=page_url,
            focus_areas=focus_areas,
            custom_notes=custom_notes,
        )

        # Step 3: Execute inference or return mock response
        if self.config.mock_mode:
            logger.info(f"[MOCK MODE] Analyzing screenshot: {image_name}")
            parsed_data = get_mock_vlm_response(
                screenshot=image_name,
                html_content=html_content,
                page_url=page_url,
            )
        else:
            logger.info(f"Sending screenshot {image_name} to Gemini model {self.config.model_name}...")
            raw_response = self._call_gemini_api(image_bytes, mime_type, prompt)
            parsed_data = self._clean_and_parse_json(raw_response)

        # Step 4: Ensure consistent top-level fields
        parsed_data["status"] = "success"
        parsed_data["page_url"] = page_url or parsed_data.get("page_url")

        # Step 5: Attach comprehensive execution metadata
        duration = round(time.time() - start_time, 2)
        parsed_data["metadata"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "screenshot_identifier": image_name,
            "page_url": page_url,
            "has_html_context": bool(html_content),
            "image_dimensions": {
                "width": image_info["width"],
                "height": image_info["height"],
            },
            "image_format": image_info["format"],
            "file_size_bytes": image_info["file_size_bytes"],
            "model": "mock-vlm-engine" if self.config.mock_mode else self.config.model_name,
            "status": "success",
            "mock_mode": self.config.mock_mode,
            "duration_seconds": duration,
        }

        return parsed_data

    def verify_fix(
        self,
        screenshot: Union[str, Path, bytes, Image.Image],
        original_issue: Dict[str, Any],
        applied_css: str,
        attempt: int = 1,
        max_attempts: int = 3,
        html_content: Optional[str] = None,
        force_status: Optional[str] = None,
        use_roi_crop: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Visually verifies whether an applied CSS fix resolved a specific UI defect.

        Week 4 Optimization: Supports ROI-targeted cropping to verify only the
        affected element rather than processing the entire screenshot, reducing token costs by >75%.

        Args:
            screenshot: Path to post-fix screenshot, bytes, or PIL Image.
            original_issue: Defect dictionary containing selector, category, description, etc.
            applied_css: The CSS rule string that was injected.
            attempt: Current attempt number.
            max_attempts: Maximum retry count.
            html_content: Optional HTML snippet of target page.
            force_status: Optional forced status for testing ('fixed', 'unresolved', 'partially_fixed').
            use_roi_crop: If True, crops to the region of interest for token-efficient verification.

        Returns:
            Structured verification result dictionary with verification_status,
            verification_reason, defect_resolved, visual_health_score, remaining_issues,
            and revised_css_suggestion.
        """
        image_bytes, mime_type, image_info, image_name = self._prepare_image_data(screenshot)

        # Mock mode / forced testing status
        if self.config.mock_mode or force_status:
            return get_mock_verification_response(
                original_issue=original_issue,
                applied_css=applied_css,
                attempt=attempt,
                max_attempts=max_attempts,
                force_status=force_status,
            )

        # If ROI cropping is requested, crop to the issue's focus area to save tokens
        target_bytes = image_bytes
        target_mime = mime_type
        if use_roi_crop:
            try:
                pil_img = Image.open(io.BytesIO(image_bytes))
                cropped_img, _ = self.optimizer.crop_for_issue(pil_img, original_issue)
                opt_bytes, opt_mime, _ = self.optimizer.optimize_image_payload(cropped_img)
                target_bytes = opt_bytes
                target_mime = opt_mime
            except Exception as e:
                logger.debug(f"ROI cropping skipped for verification: {e}")

        prompt = build_verification_prompt(
            original_issue=original_issue,
            applied_css=applied_css,
            attempt=attempt,
            max_attempts=max_attempts,
            html_content=html_content,
        )

        raw_response = self._call_gemini_api(target_bytes, target_mime, prompt)
        return self._clean_and_parse_json(raw_response)

    def analyze_optimized(
        self,
        screenshot: Union[str, Path, bytes, Image.Image],
        html_content: Optional[str] = None,
        page_url: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
        custom_notes: Optional[str] = None,
        chunk_if_tall: bool = True,
        detect_rois: bool = True,
        max_rois: int = 3,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Week 4 VLM Optimization Pipeline:
        1. Analyzes image dimensions and checks if chunking is beneficial (tall or large page).
        2. Automatically detects suspicious regions (overlapping text, clipped buttons, misaligned elements).
        3. Analyzes relevant regions of interest (ROI) instead of repeatedly processing the full page.
        4. Calculates and attaches token reduction metrics and latency benchmarks.

        Args:
            screenshot: Image path, bytes, or PIL Image.
            html_content: Optional HTML context.
            page_url: Optional page URL.
            focus_areas: Optional priority areas.
            custom_notes: Optional extra test notes.
            chunk_if_tall: If True and height > 1200, uses overlapping chunking.
            detect_rois: If True, identifies and crops suspicious UI components.
            max_rois: Maximum candidate ROIs to analyze.

        Returns:
            Structured OmniSight QA report with comprehensive optimization metadata.
        """
        tracker = PerformanceTracker()
        tracker.start()

        # Step 1: Prepare image data
        image_bytes, mime_type, image_info, image_name = self._prepare_image_data(screenshot)
        pil_img = Image.open(io.BytesIO(image_bytes))
        img_w, img_h = pil_img.size
        img_w, img_h = pil_img.size

        # Step 2: Check if image is tall enough to benefit from chunking
        if chunk_if_tall and img_h > 1200:
            logger.info(f"Screenshot height ({img_h}px) exceeds 1200px threshold -> applying image chunking.")
            chunk_report = self.analyze_chunks(
                screenshot=pil_img,
                chunk_width=1200,
                chunk_height=800,
                overlap_px=60,
                vertical_only=True,
                html_content=html_content,
                page_url=page_url,
                **kwargs,
            )
            return chunk_report

        # Step 3: Fast ROI Detection
        rois = []
        if detect_rois:
            rois = self.optimizer.crop_suspicious_regions(
                pil_img, html_content=html_content, max_regions=max_rois
            )
            logger.info(f"Detected {len(rois)} suspicious UI regions for focused VLM analysis.")

        # Step 4: Token Optimization Calculation
        opt_bytes, opt_mime, token_metrics = self.optimizer.optimize_image_payload(pil_img)
        tracker.stop_preprocessing()

        # Step 5: VLM Inference (Mock or Live)
        if self.config.mock_mode:
            parsed_data = get_mock_vlm_response(
                screenshot=image_name,
                html_content=html_content,
                page_url=page_url,
            )
            tracker.stop_inference()
        else:
            prompt = build_analysis_prompt(
                html_content=html_content,
                page_url=page_url,
                focus_areas=focus_areas,
                custom_notes=custom_notes,
            )
            raw_response = self._call_gemini_api(opt_bytes, opt_mime, prompt)
            tracker.stop_inference()
            parsed_data = self._clean_and_parse_json(raw_response)

        # Step 6: Attach optimization metrics
        perf_metrics = tracker.get_metrics(chunks_total=1, chunks_skipped_blank=0, chunks_analyzed=1)
        parsed_data["status"] = "success"
        parsed_data["page_url"] = page_url or parsed_data.get("page_url")

        parsed_data["optimization"] = {
            "token_metrics": token_metrics.to_dict(),
            "performance_metrics": perf_metrics.to_dict(),
            "detected_rois": [r.to_dict() for r in rois],
            "mode": "optimized_vlm",
        }

        # Update metadata
        parsed_data.setdefault("metadata", {})
        parsed_data["metadata"]["optimization_enabled"] = True
        parsed_data["metadata"]["tokens_saved_estimate"] = token_metrics.estimated_tokens_saved
        parsed_data["metadata"]["token_reduction_pct"] = token_metrics.token_reduction_pct
        parsed_data["metadata"]["detected_suspicious_regions_count"] = len(rois)

        return parsed_data

    def analyze_chunks(
        self,
        screenshot: Union[str, Path, bytes, Image.Image],
        chunk_width: int = 1200,
        chunk_height: int = 800,
        overlap_px: int = 60,
        vertical_only: bool = True,
        html_content: Optional[str] = None,
        page_url: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Divides screenshot into overlapping chunks and analyzes non-blank chunks.

        Skips uniform whitespace chunks, saving significant latency and tokens.
        Merges results and maps defect coordinates back to the full page.
        """
        tracker = PerformanceTracker()
        tracker.start()

        # Step 1: Chunk image
        image_bytes, mime_type, image_info, image_name = self._prepare_image_data(screenshot)
        pil_img = Image.open(io.BytesIO(image_bytes))
        img_w, img_h = pil_img.size
        chunks = self.optimizer.chunk_screenshot(pil_img, vertical_only=vertical_only)

        tracker.stop_preprocessing()

        # Step 2: Process non-blank chunks
        chunk_results = []
        analyzed_count = 0
        skipped_blank = 0

        for chunk in chunks:
            if chunk.is_blank:
                skipped_blank += 1
                logger.info(f"Skipping blank Chunk #{chunk.chunk_id} ({chunk.blank_reason}) -> 0 tokens used.")
                continue

            analyzed_count += 1
            if self.config.mock_mode:
                chunk_res = get_mock_vlm_response(
                    screenshot=f"{image_name}_chunk_{chunk.chunk_id}.png",
                    html_content=html_content,
                    page_url=page_url,
                )
            else:
                opt_bytes, opt_mime, _ = self.optimizer.optimize_image_payload(chunk.image)
                prompt = build_chunk_analysis_prompt(
                    chunk_index=chunk.chunk_id,
                    total_chunks=len(chunks),
                    bbox=chunk.bbox,
                    page_url=page_url,
                )
                raw_resp = self._call_gemini_api(opt_bytes, opt_mime, prompt)
                chunk_res = self._clean_and_parse_json(raw_resp)

            chunk_results.append(chunk_res)

        tracker.stop_inference()

        # Step 3: Merge results across chunks
        merged_report = self.optimizer.chunker.merge_chunk_results(chunk_results, chunks)

        # Step 4: Token and Performance calculations
        full_tokens = VLMTokenOptimizer.estimate_tokens(pil_img.width, pil_img.height)
        used_tokens = analyzed_count * 258
        saved_tokens = max(0, full_tokens - used_tokens)
        token_red_pct = ((full_tokens - used_tokens) / full_tokens * 100.0) if full_tokens > 0 else 0.0

        perf_metrics = tracker.get_metrics(
            chunks_total=len(chunks),
            chunks_skipped_blank=skipped_blank,
            chunks_analyzed=analyzed_count,
        )

        merged_report["page_url"] = page_url
        merged_report["optimization"] = {
            "chunk_metrics": {
                "chunks_total": len(chunks),
                "chunks_analyzed": analyzed_count,
                "chunks_skipped_blank": skipped_blank,
                "overlap_px": overlap_px,
                "chunks": [c.to_dict() for c in chunks],
            },
            "token_metrics": {
                "estimated_tokens_original_full": full_tokens,
                "estimated_tokens_chunked_total": used_tokens,
                "estimated_tokens_saved": saved_tokens,
                "token_reduction_pct": round(token_red_pct, 1),
            },
            "performance_metrics": perf_metrics.to_dict(),
            "mode": "chunked_vlm",
        }

        merged_report["metadata"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "screenshot_identifier": image_name,
            "page_url": page_url,
            "model": "mock-vlm-engine" if self.config.mock_mode else self.config.model_name,
            "status": "success",
            "mock_mode": self.config.mock_mode,
            "duration_seconds": round(perf_metrics.total_time_ms / 1000.0, 2),
            "chunks_total": len(chunks),
            "chunks_skipped_blank": skipped_blank,
            "tokens_saved_estimate": saved_tokens,
        }

        return merged_report

    def analyze_roi(
        self,
        screenshot: Union[str, Path, bytes, Image.Image],
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
        roi_label: str = "suspicious_region",
        issue: Optional[Dict[str, Any]] = None,
        html_content: Optional[str] = None,
        padding: int = 30,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Analyzes a single cropped region of interest (ROI) with minimal token overhead.
        """
        tracker = PerformanceTracker()
        tracker.start()

        image_bytes, mime_type, image_info, image_name = self._prepare_image_data(screenshot)
        pil_img = Image.open(io.BytesIO(image_bytes))
        img_w, img_h = pil_img.size

        # Crop using issue or bbox
        if issue:
            cropped_img, effective_bbox = self.optimizer.crop_for_issue(pil_img, issue, padding=padding)
            label = issue.get("category", roi_label)
        elif roi_bbox:
            cropped_img = self.optimizer.cropper.crop_region(pil_img, roi_bbox, padding=padding)
            effective_bbox = roi_bbox
            label = roi_label
        else:
            candidates = self.optimizer.crop_suspicious_regions(pil_img, html_content=html_content, max_regions=1)
            candidate = candidates[0]
            cropped_img = candidate.cropped_image or pil_img
            effective_bbox = candidate.bbox
            label = candidate.label

        # Optimize cropped payload relative to full screenshot baseline
        opt_bytes, opt_mime, token_metrics = self.optimizer.optimize_image_payload(
            cropped_img, baseline_dimensions=(img_w, img_h)
        )
        tracker.stop_preprocessing()

        # VLM Inference
        if self.config.mock_mode:
            result = get_mock_roi_response(roi_label=label, bbox=effective_bbox)
            tracker.stop_inference()
        else:
            prompt = build_roi_analysis_prompt(
                roi_label=label,
                roi_bbox=effective_bbox,
                html_snippet=html_content,
            )
            raw_resp = self._call_gemini_api(opt_bytes, opt_mime, prompt)
            tracker.stop_inference()
            result = self._clean_and_parse_json(raw_resp)

        perf = tracker.get_metrics(chunks_total=1, chunks_skipped_blank=0, chunks_analyzed=1)
        result["status"] = "success"
        result["optimization"] = {
            "roi_bbox": list(effective_bbox),
            "roi_label": label,
            "token_metrics": token_metrics.to_dict(),
            "performance_metrics": perf.to_dict(),
            "mode": "roi_crop",
        }
        return result


def analyze_screenshot(
    screenshot: Union[str, Path, bytes, Image.Image],
    html_content: Optional[str] = None,
    page_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    mock_mode: Optional[bool] = None,
    optimize: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to analyze a screenshot in a single call.

    Args:
        screenshot: Path to image, bytes, or PIL Image.
        html_content: Optional HTML snippet or full DOM.
        page_url: Optional page URL.
        api_key: Optional Gemini API key (defaults to GEMINI_API_KEY env var).
        model_name: Optional Gemini model name.
        mock_mode: Optional boolean to enable mock mode.
        optimize: If True, uses Week 4 VLM optimization pipeline.
        **kwargs: Additional parameters passed to analyze_screenshot.

    Returns:
        Structured dictionary conforming to the OmniSight VLM QA schema.
    """
    analyzer = VisionAnalyzer(
        api_key=api_key,
        model_name=model_name,
        mock_mode=mock_mode,
    )
    if optimize:
        return analyzer.analyze_optimized(
            screenshot=screenshot,
            html_content=html_content,
            page_url=page_url,
            **kwargs,
        )
    return analyzer.analyze_screenshot(
        screenshot=screenshot,
        html_content=html_content,
        page_url=page_url,
        **kwargs,
    )


def verify_fix(
    screenshot: Union[str, Path, bytes, Image.Image],
    original_issue: Dict[str, Any],
    applied_css: str,
    attempt: int = 1,
    max_attempts: int = 3,
    html_content: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    mock_mode: Optional[bool] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to verify a CSS fix against a screenshot in a single call.
    """
    analyzer = VisionAnalyzer(
        api_key=api_key,
        model_name=model_name,
        mock_mode=mock_mode,
    )
    return analyzer.verify_fix(
        screenshot=screenshot,
        original_issue=original_issue,
        applied_css=applied_css,
        attempt=attempt,
        max_attempts=max_attempts,
        html_content=html_content,
        **kwargs,
    )


def analyze_optimized(
    screenshot: Union[str, Path, bytes, Image.Image],
    html_content: Optional[str] = None,
    page_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    mock_mode: Optional[bool] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to run Week 4 optimized VLM analysis.
    """
    analyzer = VisionAnalyzer(
        api_key=api_key,
        model_name=model_name,
        mock_mode=mock_mode,
    )
    return analyzer.analyze_optimized(
        screenshot=screenshot,
        html_content=html_content,
        page_url=page_url,
        **kwargs,
    )

