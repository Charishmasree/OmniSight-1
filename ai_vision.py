"""
OmniSight - Member 2: AI/VLM Basic Setup and Prompt Testing
AI Vision Module (ai_vision.py)

Core engine for analyzing website screenshots using Vision-Language Models (VLM).
Integrates Google Gemini VLM API, handles screenshot validation, error handling,
mock inference for testing, and returns structured QA findings.
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

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
from prompts import build_analysis_prompt, get_mock_analysis_data
from result_logger import save_analysis_report

# Configure module logger
logger = logging.getLogger("omnisight.ai_vision")


class VisionAnalyzer:
    """
    Vision-Language Model (VLM) analyzer for automated website screenshot QA.

    Accepts screenshots from Playwright browser automation (Member 1) and
    produces structured JSON analysis ready for FastAPI backend (Member 3).
    """

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
            api_key: Optional Gemini API key (defaults to GEMINI_API_KEY env var).
            model_name: Model identifier (e.g. 'gemini-2.5-flash').
            mock_mode: When True, bypasses API calls and returns synthetic QA findings.
            config: Optional pre-configured VisionConfig instance.
        """
        overrides = {}
        if api_key is not None:
            overrides["gemini_api_key"] = api_key
        if model_name is not None:
            overrides["model_name"] = model_name
        if mock_mode is not None:
            overrides["mock_mode"] = mock_mode

        self.config = config or get_config(**overrides)
        self._client = None

        # Validate configuration if not in mock mode
        if not self.config.mock_mode:
            self.config.validate()
            self._init_client()
        else:
            logger.info("VisionAnalyzer initialized in MOCK MODE (no external API calls will be made).")

    def _init_client(self) -> None:
        """Initializes the google.genai Client."""
        try:
            from google import genai
            self._client = genai.Client(api_key=self.config.gemini_api_key)
            logger.info(f"Initialized Google GenAI Client with model: {self.config.model_name}")
        except Exception as e:
            raise ModelInferenceError(f"Failed to initialize Google GenAI Client: {e}") from e

    def validate_image(self, image_path: Union[str, Path]) -> Path:
        """
        Validates the screenshot file path, existence, extension, size, and image integrity.

        Args:
            image_path: Path to the screenshot image file.

        Returns:
            Resolved Path object if valid.

        Raises:
            ImageNotFoundError: If the file does not exist.
            InvalidImageFormatError: If the extension is not supported.
            ImageSizeExceededError: If the file exceeds max_image_size_mb.
            ImageCorruptedError: If the image is corrupted or unreadable.
        """
        path = Path(image_path).resolve()

        # 1. Existence check
        if not path.exists():
            raise ImageNotFoundError(f"Screenshot file does not exist: {path}")
        if not path.is_file():
            raise ImageNotFoundError(f"Screenshot path is not a file: {path}")

        # 2. Extension check
        ext = path.suffix.lower()
        if ext not in self.config.supported_formats:
            raise InvalidImageFormatError(
                f"Unsupported image extension '{ext}'. "
                f"Supported formats: {sorted(list(self.config.supported_formats))}"
            )

        # 3. File size check
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.config.max_image_size_mb:
            raise ImageSizeExceededError(
                f"Image size ({size_mb:.2f} MB) exceeds maximum allowed limit "
                f"of {self.config.max_image_size_mb} MB."
            )

        # 4. Integrity check via PIL
        try:
            with Image.open(path) as img:
                img.verify()
        except (UnidentifiedImageError, OSError, Exception) as e:
            raise ImageCorruptedError(f"Corrupted or invalid image file: {path}. Details: {e}") from e

        return path

    def get_image_info(self, image_path: Path) -> Dict[str, Any]:
        """Reads width, height, format, and file size of a validated image."""
        with Image.open(image_path) as img:
            width, height = img.size
            img_format = img.format or image_path.suffix.lstrip(".").upper()
        return {
            "width": width,
            "height": height,
            "format": img_format,
            "file_size_bytes": image_path.stat().st_size,
        }

    def _clean_and_parse_json(self, raw_text: str) -> Dict[str, Any]:
        """
        Strips markdown code fences and parses raw model output as JSON.

        Args:
            raw_text: Raw string returned by the VLM.

        Returns:
            Parsed Python dictionary.

        Raises:
            ResponseParseError: If parsing fails or required keys are missing.
        """
        cleaned = raw_text.strip()

        # Remove markdown code block markers (e.g. ```json ... ```)
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from VLM response. Raw response: {raw_text[:200]}...")
            raise ResponseParseError(
                f"Model response was not valid JSON: {e}. Raw content preview: {cleaned[:150]}"
            ) from e

        if not isinstance(data, dict):
            raise ResponseParseError(f"Expected JSON object (dict), but received: {type(data).__name__}")

        # Ensure essential top-level keys exist (provide defaults if model omitted any)
        data.setdefault("page_summary", "Page visual analysis completed.")
        data.setdefault("visual_health_score", 80)
        data.setdefault("ui_elements", {})
        data.setdefault("visual_issues", [])
        data.setdefault("recommendations", [])

        return data

    def _call_gemini_api(self, image_path: Path, prompt_text: str) -> str:
        """
        Sends the screenshot and prompt to Google Gemini via google.genai.

        Args:
            image_path: Validated path to screenshot.
            prompt_text: Formatted prompt instructions.

        Returns:
            Raw response text from model.

        Raises:
            ModelInferenceError: If network, auth, or model call fails.
        """
        from google.genai import types

        ext = image_path.suffix.lower()
        mime_type = self.MIME_TYPES.get(ext, "image/png")

        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

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
            logger.error(f"Error during VLM inference: {e}")
            raise ModelInferenceError(f"VLM inference failed: {e}") from e

    def analyze_screenshot(
        self,
        image_path: Union[str, Path],
        focus_areas: Optional[List[str]] = None,
        custom_notes: Optional[str] = None,
        custom_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes a website screenshot and returns a structured dictionary of UI elements and issues.

        Args:
            image_path: Path to screenshot generated by Member 1 (Playwright).
            focus_areas: Optional list of areas to focus on (e.g. ['buttons', 'overlapping_elements']).
            custom_notes: Optional extra test notes.
            custom_prompt: Optional full override for prompt text.

        Returns:
            Structured dictionary with metadata, UI elements, visual issues, and health score.

        Raises:
            ImageNotFoundError, InvalidImageFormatError, ImageCorruptedError,
            ModelInferenceError, ResponseParseError, APIKeyMissingError.
        """
        start_time = time.time()

        # Step 1: Validate screenshot path & integrity
        validated_path = self.validate_image(image_path)
        image_info = self.get_image_info(validated_path)

        # Step 2: Prepare prompt
        prompt = custom_prompt or build_analysis_prompt(
            focus_areas=focus_areas,
            custom_notes=custom_notes,
        )

        # Step 3: Execute inference or mock response
        if self.config.mock_mode:
            logger.info(f"[MOCK MODE] Analyzing screenshot: {validated_path.name}")
            parsed_data = get_mock_analysis_data(screenshot_path=str(validated_path))
        else:
            logger.info(f"Sending screenshot {validated_path.name} to {self.config.model_name}...")
            raw_response = self._call_gemini_api(validated_path, prompt)
            parsed_data = self._clean_and_parse_json(raw_response)

        # Step 4: Attach comprehensive metadata
        duration = round(time.time() - start_time, 2)
        parsed_data["metadata"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "screenshot_path": str(validated_path),
            "screenshot_filename": validated_path.name,
            "image_dimensions": {
                "width": image_info["width"],
                "height": image_info["height"],
            },
            "file_size_bytes": image_info["file_size_bytes"],
            "model": "mock-vlm-engine" if self.config.mock_mode else self.config.model_name,
            "status": "success",
            "mock_mode": self.config.mock_mode,
            "duration_seconds": duration,
        }

        return parsed_data


def analyze_and_save(
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None,
    analyzer: Optional[VisionAnalyzer] = None,
    **analyzer_kwargs,
) -> Tuple[Dict[str, Any], Path]:
    """
    High-level convenience function: analyzes a screenshot and saves the report to JSON.

    Args:
        image_path: Path to the screenshot image.
        output_path: Specific destination JSON path.
        output_dir: Target folder to save the report in.
        filename: Custom JSON filename.
        analyzer: Optional pre-existing VisionAnalyzer instance.
        **analyzer_kwargs: Arguments passed to VisionAnalyzer if creating a new instance.

    Returns:
        Tuple of (analysis_dictionary, saved_json_path).
    """
    active_analyzer = analyzer or VisionAnalyzer(**analyzer_kwargs)
    result = active_analyzer.analyze_screenshot(image_path)
    saved_path = save_analysis_report(
        data=result,
        output_path=output_path,
        output_dir=output_dir,
        filename=filename,
    )
    return result, saved_path
