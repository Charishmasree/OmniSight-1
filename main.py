"""
OmniSight - Member 2: AI/VLM Basic Setup and Prompt Testing
Command-Line Interface & Demonstration Script (main.py)

Provides an easy-to-use CLI to run visual QA analysis on any screenshot,
support mock mode for zero-cost testing, and generate sample reports.
"""

import argparse
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from ai_vision import VisionAnalyzer, analyze_and_save
from result_logger import print_summary, save_analysis_report
from config import OmniSightVisionError, get_config


def create_sample_screenshot(output_path: Path = Path("sample_screenshots/sample_ui.png")) -> Path:
    """
    Generates a realistic synthetic website screenshot for testing
    when no browser screenshots are available yet.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1200, 800
    # Create image with clean dark/light UI palette
    image = Image.new("RGB", (width, height), color=(245, 247, 250))
    draw = ImageDraw.Draw(image)

    # Navbar background
    draw.rectangle([(0, 0), (width, 70)], fill=(20, 24, 33))

    # Brand Logo
    draw.rectangle([(40, 20), (160, 50)], fill=(79, 70, 229))
    draw.text((60, 26), "OMNISIGHT", fill=(255, 255, 255))

    # Nav links
    nav_items = ["Dashboard", "Test Suites", "Analytics", "Settings"]
    for i, item in enumerate(nav_items):
        draw.text((220 + (i * 120), 28), item, fill=(203, 213, 225))

    # Hero section
    draw.rectangle([(40, 110), (1160, 320)], fill=(255, 255, 255), outline=(226, 232, 240), width=2)
    draw.text((70, 140), "Automated UI Quality & Visual Regression Suite", fill=(15, 23, 42))
    draw.text((70, 180), "Analyze website screenshots for overlapping elements, contrast issues, and layout bugs.", fill=(100, 116, 139))

    # Intentionally draw an overlapping text bug over a CTA button to simulate real-world defect!
    draw.rectangle([(70, 230), (220, 280)], fill=(79, 70, 229))  # Primary CTA
    draw.text((105, 245), "Run Test", fill=(255, 255, 255))

    # Overlapping text defect
    draw.text((140, 260), "OVERLAPPING BADGE DEFECT", fill=(239, 68, 68))

    # Card grid
    for col in range(3):
        x1 = 40 + (col * 385)
        x2 = x1 + 350
        draw.rectangle([(x1, 350), (x2, 650)], fill=(255, 255, 255), outline=(226, 232, 240), width=1)
        draw.text((x1 + 20, 370), f"Card Component #{col + 1}", fill=(30, 41, 59))
        draw.rectangle([(x1 + 20, 410), (x2 - 20, 550)], fill=(241, 245, 249))

        # Card button
        draw.rectangle([(x1 + 20, 580), (x2 - 20, 620)], fill=(243, 244, 246), outline=(209, 213, 219), width=1)
        draw.text((x1 + 100, 592), "Inspect Details", fill=(55, 65, 81))

    # Footer with very low contrast disclaimer (simulating contrast issue)
    draw.rectangle([(0, 730), (width, height)], fill=(255, 255, 255))
    draw.text((40, 755), "© 2026 OmniSight Technologies. Unreadable low contrast disclaimer text.", fill=(225, 228, 232))

    image.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="OmniSight AI/VLM Screenshot Analyzer (Member 2 Module)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in mock mode on sample screenshot (no API key required):
  python main.py --mock

  # Run real VLM inference on a Playwright screenshot:
  python main.py path/to/playwright_screenshot.png

  # Generate sample test screenshot only:
  python main.py --create-sample
        """
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help="Path to the screenshot image file (PNG, JPG, WEBP)."
    )
    parser.add_argument(
        "--mock", "-m",
        action="store_true",
        help="Run in mock mode (uses realistic synthetic VLM responses, no API key needed)."
    )
    parser.add_argument(
        "--key", "-k",
        type=str,
        default=None,
        help="Gemini API Key (overrides GEMINI_API_KEY environment variable)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="VLM model name (default: gemini-2.5-flash)."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory to save JSON analysis report (default: analysis_reports/)."
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Generate a synthetic test screenshot at sample_screenshots/sample_ui.png and exit."
    )

    args = parser.parse_args()

    # Handle sample screenshot creation request
    if args.create_sample:
        created = create_sample_screenshot()
        print(f"Sample screenshot created at: {created.resolve()}")
        return 0

    # Determine image path
    target_image = args.image_path
    if not target_image:
        sample_path = Path("sample_screenshots/sample_ui.png")
        if not sample_path.exists():
            print("No screenshot specified. Generating a test screenshot at sample_screenshots/sample_ui.png...")
            create_sample_screenshot(sample_path)
        target_image = str(sample_path)
        # Default to mock mode if no API key is set and no image was specified
        config = get_config()
        if not args.mock and not config.gemini_api_key and not args.key:
            print("No GEMINI_API_KEY detected. Automatically running in --mock mode for testing.")
            args.mock = True

    try:
        analyzer = VisionAnalyzer(
            api_key=args.key,
            model_name=args.model,
            mock_mode=args.mock,
        )

        print(f"\n[OmniSight] Analyzing screenshot: {target_image}")
        print(f"[OmniSight] Mode: {'MOCK (Synthetic)' if analyzer.config.mock_mode else f'LIVE VLM ({analyzer.config.model_name})'}")

        results, saved_path = analyze_and_save(
            image_path=target_image,
            output_dir=args.output_dir,
            analyzer=analyzer,
        )

        print_summary(results)
        print(f"[SUCCESS] JSON Report saved to: {saved_path.resolve()}\n")
        return 0

    except OmniSightVisionError as e:
        print(f"\n[ERROR] OmniSight Vision Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
