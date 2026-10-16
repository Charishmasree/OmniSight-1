"""
OmniSight - Member 2: AI/VLM Basic Setup and Prompt Testing
Configuration Module (config.py)

Manages environment variables, default settings, and custom exceptions.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Set
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()


# ============================================================================
# Custom Exception Hierarchy
# ============================================================================

class OmniSightVisionError(Exception):
    """Base exception for all OmniSight Vision module errors."""
    pass


class APIKeyMissingError(OmniSightVisionError):
    """Raised when an external VLM API key is required but not provided."""
    pass


class ImageNotFoundError(OmniSightVisionError):
    """Raised when the specified screenshot file cannot be found."""
    pass


class InvalidImageFormatError(OmniSightVisionError):
    """Raised when an image format is unsupported or file extension is invalid."""
    pass


class ImageCorruptedError(InvalidImageFormatError):
    """Raised when the image file cannot be decoded or is corrupted."""
    pass


class ImageSizeExceededError(InvalidImageFormatError):
    """Raised when the image exceeds the maximum allowed file size."""
    pass


class ModelInferenceError(OmniSightVisionError):
    """Raised when the VLM fails to process the image or returns an API error."""
    pass


class ResponseParseError(OmniSightVisionError):
    """Raised when the VLM output cannot be parsed into the expected JSON structure."""
    pass


# ============================================================================
# Configuration Data Class
# ============================================================================

@dataclass
class VisionConfig:
    """
    Central configuration for the Vision-Language Model (VLM) analyzer.

    Reads from environment variables with sensible defaults for quick testing.
    """
    # API Credentials
    gemini_api_key: str = field(
        default_factory=lambda: (
            os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        )
    )

    # VLM Model Selection (gemini-2.5-flash / gemini-2.0-flash are state-of-the-art for vision QA)
    model_name: str = field(
        default_factory=lambda: os.getenv("VLM_MODEL_NAME", "gemini-2.5-flash")
    )

    # Mock Mode: when True, produces deterministic QA mock output without API calls
    mock_mode: bool = field(
        default_factory=lambda: os.getenv("VLM_MOCK_MODE", "false").lower() in ("true", "1", "yes")
    )

    # Output directory for JSON reports
    output_dir: Path = field(
        default_factory=lambda: Path(os.getenv("VLM_OUTPUT_DIR", "analysis_reports"))
    )

    # Supported image extensions for screenshots
    supported_formats: Set[str] = field(
        default_factory=lambda: {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    )

    # Maximum image file size in Megabytes
    max_image_size_mb: float = 15.0

    # API Request timeout in seconds
    request_timeout_seconds: int = 60

    # Temperature for deterministic, reproducible QA responses
    temperature: float = 0.2

    def validate(self) -> None:
        """
        Validate critical configuration parameters.
        Raises APIKeyMissingError if real inference is requested but no key is provided.
        """
        if not self.mock_mode and not self.gemini_api_key:
            raise APIKeyMissingError(
                "GEMINI_API_KEY is not set. Please add it to your .env file, "
                "set the GEMINI_API_KEY environment variable, or enable mock mode "
                "(VLM_MOCK_MODE=true) for offline testing."
            )


def get_config(**overrides) -> VisionConfig:
    """
    Factory function to get a validated VisionConfig instance,
    allowing optional keyword overrides.
    """
    config = VisionConfig(**overrides)
    return config
