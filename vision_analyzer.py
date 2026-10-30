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
    ) -> Dict[str, Any]:
        """
        Week 3: Sends an updated screenshot to the VLM to verify if an applied CSS fix
        resolved the original visual defect.

        Args:
            screenshot: Path to post-fix screenshot, raw bytes, or PIL Image.
            original_issue: Issue dictionary detected in prior analysis.
            applied_css: CSS rules injected into the DOM.
            attempt: Current retry attempt index (1-based).
            max_attempts: Maximum retry attempts configured.
            html_content: Optional DOM context.
            force_status: Optional forced verification status for deterministic testing.

        Returns:
            Structured verification result dict conforming to the verification schema:
            - verification_status: 'fixed' | 'partially_fixed' | 'unresolved'
            - verification_reason: str
            - defect_resolved: bool
            - visual_health_score: int
            - remaining_issues: list
            - revised_css_suggestion: Optional dict
            - metadata: execution timestamp, model, duration
        """
        start_time = time.time()
        image_bytes, mime_type, image_info, image_name = self._prepare_image_data(screenshot)

        if self.config.mock_mode:
            logger.info(f"[MOCK MODE] Verifying CSS fix for attempt {attempt}/{max_attempts} on {image_name}")
            parsed_data = get_mock_verification_response(
                original_issue=original_issue,
                applied_css=applied_css,
                attempt=attempt,
                max_attempts=max_attempts,
                force_status=force_status,
            )
        else:
            prompt = build_verification_prompt(
                original_issue=original_issue,
                applied_css=applied_css,
                attempt=attempt,
                max_attempts=max_attempts,
                html_content=html_content,
            )
            logger.info(f"Sending verification screenshot {image_name} to {self.config.model_name}...")
            raw_response = self._call_gemini_api(image_bytes, mime_type, prompt)
            parsed_data = self._clean_and_parse_json(raw_response)

        # Normalize verification fields
        status = str(parsed_data.get("verification_status", "unresolved")).lower()
        if status not in ("fixed", "partially_fixed", "unresolved"):
            if "fix" in status or "resolv" in status or "pass" in status:
                status = "fixed"
            elif "part" in status:
                status = "partially_fixed"
            else:
                status = "unresolved"

        parsed_data["verification_status"] = status
        parsed_data.setdefault("defect_resolved", (status == "fixed"))
        parsed_data.setdefault("verification_reason", "Verification completed.")
        parsed_data.setdefault("visual_health_score", 95 if status == "fixed" else 65)
        parsed_data.setdefault("remaining_issues", [])
        parsed_data.setdefault("revised_css_suggestion", None)

        duration = round(time.time() - start_time, 2)
        parsed_data["metadata"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "screenshot_identifier": image_name,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "model": "mock-vlm-engine" if self.config.mock_mode else self.config.model_name,
            "mock_mode": self.config.mock_mode,
            "duration_seconds": duration,
        }
        return parsed_data


def analyze_screenshot(
    screenshot: Union[str, Path, bytes, Image.Image],
    html_content: Optional[str] = None,
    page_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    mock_mode: Optional[bool] = None,
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
        **kwargs: Additional parameters passed to analyze_screenshot.

    Returns:
        Structured dictionary conforming to the OmniSight VLM QA schema.
    """
    analyzer = VisionAnalyzer(
        api_key=api_key,
        model_name=model_name,
        mock_mode=mock_mode,
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

