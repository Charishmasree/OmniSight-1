"""
OmniSight - Member 2: Week 4 VLM Optimization Demonstration Script
Run this script directly in VS Code (F5, 'Run Python File', or via Terminal).

Demonstrates Week 4 Member 2 core responsibilities:
1. Image Chunking: Slicing tall/large webpage screenshots into overlapping sections
   with boundary safety and blank-chunk skipping.
2. Image Cropping: Automatic detection and cropping of suspicious UI defect components
   (overlapping text, clipped buttons, misaligned elements).
3. VLM Optimization: Analyzing only relevant screenshot regions instead of repeatedly
   processing the entire full-page screenshot.
4. Token Reduction: Lowering multimodal input tokens and payload bytes by 75%+ while
   maintaining defect fidelity.
5. Latency Improvement: Eliminating wasteful inference on blank whitespace and benchmarking speedups.
"""

import json
import os
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.optimizer import (
    VLMOptimizer,
    ImageChunker,
    ImageCropper,
    VLMTokenOptimizer,
    PerformanceTracker,
)
from ai.vision_analyzer import VisionAnalyzer, analyze_optimized


def print_banner(title: str, char: str = "="):
    width = 75
    print("\n" + char * width)
    print(f" {title.center(width - 2)} ")
    print(char * width)


def create_demo_tall_screenshot(output_path: Path) -> Path:
    """
    Generates a realistic tall 1200x2400 multi-section webpage screenshot
    with intentional UI defects across different sections and a blank bottom section.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1200, 2400

    img = Image.new("RGB", (width, height), color=(248, 250, 252))
    draw = ImageDraw.Draw(img)

    # -------------------------------------------------------------------------
    # Section 1: Top Navigation & Hero Banner (Y: 0 - 800)
    # -------------------------------------------------------------------------
    # Navbar
    draw.rectangle([(0, 0), (width, 70)], fill=(15, 23, 42))
    draw.rectangle([(40, 20), (160, 50)], fill=(99, 102, 241))
    draw.text((60, 27), "OMNISIGHT", fill=(255, 255, 255))
    nav_links = ["Dashboard", "Analytics", "Audits", "Settings"]
    for i, item in enumerate(nav_links):
        draw.text((220 + i * 130, 28), item, fill=(203, 213, 225))

    # Hero Banner
    draw.rectangle([(40, 110), (1160, 340)], fill=(255, 255, 255), outline=(226, 232, 240), width=2)
    draw.text((70, 140), "Enterprise AI Vision QA & Self-Healing Agent", fill=(15, 23, 42))
    draw.text((70, 180), "Automated multimodal inspection for modern web applications.", fill=(100, 116, 139))

    # Defect #1: Overlapping Subtitle Text over CTA Button
    draw.rectangle([(70, 230), (230, 280)], fill=(99, 102, 241))
    draw.text((100, 247), "Run Audit", fill=(255, 255, 255))
    draw.text((130, 260), "OVERLAPPING BADGE DEFECT", fill=(239, 68, 68))

    # Defect #2: Misaligned Secondary Button
    draw.rectangle([(250, 238), (410, 288)], fill=(241, 245, 249), outline=(203, 213, 225), width=1)
    draw.text((280, 255), "Learn More", fill=(51, 65, 85))

    # -------------------------------------------------------------------------
    # Section 2: Metrics Grid & Analytics (Y: 800 - 1600)
    # -------------------------------------------------------------------------
    draw.rectangle([(0, 800), (width, 810)], fill=(226, 232, 240))  # Section divider
    draw.text((40, 840), "Performance Analytics & Defect Metrics", fill=(30, 41, 59))

    # Three metric cards
    for col in range(3):
        x1 = 40 + (col * 385)
        x2 = x1 + 350
        draw.rectangle([(x1, 880), (x2, 1200)], fill=(255, 255, 255), outline=(226, 232, 240), width=1)
        draw.text((x1 + 20, 900), f"Metric Card #{col + 1}", fill=(30, 41, 59))

        # Defect #3 in Card 2: Clipped Text container without ellipsis
        if col == 1:
            draw.rectangle([(x1 + 20, 930), (x2 - 10, 980)], fill=(254, 242, 242))
            draw.text((x1 + 25, 945), "Total Converted Registrations Q4 (Clipped Harshly)", fill=(220, 38, 38))
        else:
            draw.rectangle([(x1 + 20, 930), (x2 - 20, 1050)], fill=(241, 245, 249))

    # -------------------------------------------------------------------------
    # Section 3: Bottom Space (Y: 1600 - 2400)
    # Uniform flat whitespace simulating large empty page buffer / spacer
    # -------------------------------------------------------------------------
    # Leaves Y: 1600 - 2400 completely blank flat color to test blank chunk skipping!

    img.save(output_path, format="PNG")
    return output_path


def run_week4_optimization_demo():
    print_banner("OmniSight - Week 4 Member 2: VLM Optimization Demonstration")
    print("Core Capabilities: Image Chunking | Image Cropping | Token Reduction | Latency Improvement")

    output_dir = PROJECT_ROOT / "analysis_reports" / "week4_optimization"
    crops_dir = output_dir / "crops"
    chunks_dir = output_dir / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Create or locate tall sample screenshot
    screenshot_path = output_dir / "tall_webpage_sample.png"
    print(f"\n[1] Generating 1200x2400 tall screenshot with multi-section defects:")
    create_demo_tall_screenshot(screenshot_path)
    print(f"    Saved: {screenshot_path.relative_to(PROJECT_ROOT)}")

    # Initialize Week 4 Optimizer
    optimizer = VLMOptimizer(chunk_width=1200, chunk_height=800, overlap_px=60)
    source_img = Image.open(screenshot_path)
    img_w, img_h = source_img.size

    # =========================================================================
    # Step 2: Feature 1 - Image Chunking with Blank Skipping
    # =========================================================================
    print_banner("Feature 1: Image Chunking & Overlap Safety", char="-")
    print(f"Target Dimensions: {img_w}w x {img_h}h pixels")
    print(f"Chunk Config:      1200w x 800h pixels | Overlap: 60px | Vertical Slicing: True\n")

    chunks = optimizer.chunk_screenshot(source_img, vertical_only=True)
    print(f"Chunk Breakdown ({len(chunks)} total slices):")

    analyzed_chunks = []
    skipped_chunks = []

    for chunk in chunks:
        status_tag = "[BLANK - SKIPPED]" if chunk.is_blank else "[CONTENT - ACTIVE]"
        print(f"  Chunk #{chunk.chunk_id}: Bounding Box = {chunk.bbox} -> {status_tag}")
        if chunk.is_blank:
            skipped_chunks.append(chunk)
            print(f"             Reason: {chunk.blank_reason} (0 VLM tokens consumed!)")
        else:
            analyzed_chunks.append(chunk)
            # Save chunk image for inspection
            chunk_file = chunks_dir / f"chunk_{chunk.chunk_id}.png"
            chunk.image.save(chunk_file)

    print(f"\n  * Summary: {len(analyzed_chunks)} active content chunks, {len(skipped_chunks)} blank chunks skipped.")
    print(f"  * Overlap Safety: 60px margin ensures boundary UI elements are not cut off.")

    # =========================================================================
    # Step 3: Feature 2 - Automated Image Cropping & Suspicious Region Detection
    # =========================================================================
    print_banner("Feature 2: Automated Cropping for Suspicious UI Defects", char="-")
    print("Scanning screenshot for UI anomalies (overlapping text, clipped buttons, misaligned elements)...")

    html_context = """
    <p class="hero-subtitle" style="margin-bottom: -20px;">Overlapping subtitle defect</p>
    <button class="cta-btn">Run Audit</button>
    <div class="metric-card" style="overflow: hidden;"><h3 class="metric-card-title">Clipped Text</h3></div>
    """

    candidates = optimizer.crop_suspicious_regions(source_img, html_content=html_context, max_regions=3)
    print(f"\nDiscovered {len(candidates)} high-confidence defect regions of interest (ROI):")

    for i, roi in enumerate(candidates, start=1):
        crop_path = crops_dir / f"{roi.label}_crop.png"
        if roi.cropped_image:
            roi.cropped_image.save(crop_path)

        print(f"\n  [ROI #{i}] {roi.label.upper()} (Confidence: {int(roi.confidence * 100)}%)")
        print(f"    - Bounding Box: {roi.bbox}")
        print(f"    - Description:  {roi.description}")
        print(f"    - Selector:     {roi.selector_hint}")
        print(f"    - Saved Crop:   {crop_path.name} ({roi.cropped_image.width}x{roi.cropped_image.height}px)")
        print(f"    - Estimated VLM Tokens: {roi.estimated_tokens} tokens")

    # =========================================================================
    # Step 4: Feature 3 & 4 - Token Reduction & Latency Benchmarks
    # =========================================================================
    print_banner("Features 3 & 4: Token Reduction & Latency Benchmarks", char="-")

    full_page_tokens = VLMTokenOptimizer.estimate_tokens(img_w, img_h)
    full_page_pixels = img_w * img_h

    # Focus on the first cropped ROI (overlapping text)
    primary_crop = candidates[0].cropped_image
    _, _, token_metrics = optimizer.optimize_image_payload(
        primary_crop, baseline_dimensions=(img_w, img_h)
    )

    print(f"Comparison: Full 1200x2400 Page vs. Focused Cropped Component")
    print(f"  * Baseline Image Dimensions:  {img_w} x {img_h} px ({full_page_pixels:,} total pixels)")
    print(f"  * Cropped Component Dimensions: {primary_crop.width} x {primary_crop.height} px ({primary_crop.width * primary_crop.height:,} pixels)")
    print(f"  * Pixel Reduction:            {token_metrics.pixel_reduction_pct:.1f}% reduction")
    print(f"  * Full Page Token Estimate:   {full_page_tokens:,} tokens")
    print(f"  * Cropped Region Tokens:      {token_metrics.estimated_tokens_processed} tokens")
    print(f"  * Token Savings:              {token_metrics.estimated_tokens_saved:,} tokens SAVED ({token_metrics.token_reduction_pct:.1f}% reduction!)")
    print(f"  * Byte Compression:           {token_metrics.byte_reduction_pct:.1f}% smaller payload")

    # =========================================================================
    # Step 5: End-to-End VLM Optimization Pipeline Execution
    # =========================================================================
    print_banner("End-to-End Execution: analyze_optimized Pipeline", char="-")
    analyzer = VisionAnalyzer(mock_mode=True)

    start_t = time.perf_counter()
    report = analyzer.analyze_optimized(
        screenshot=screenshot_path,
        html_content=html_context,
        page_url="https://app.omnisight.local/dashboard",
        chunk_if_tall=True,
    )
    duration_s = time.perf_counter() - start_t

    print(f"Status:              {report.get('status').upper()}")
    print(f"Visual Health Score: {report.get('visual_health_score')}/100")
    print(f"Total Defects Found: {len(report.get('issues', []))}")
    print(f"Execution Latency:   {duration_s:.3f}s (Mock Mode)")

    # Save standardized JSON report
    report_file = output_dir / "week4_optimization_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[SUCCESS] Comprehensive Week 4 Optimization Report saved to:")
    print(f"          {report_file.resolve()}")

    print_banner("Week 4 Member 2 Demonstration Complete!", char="=")
    print("\nTo run the automated Week 4 test suite, execute:")
    print("  pytest tests/test_vlm_optimization.py -v\n")


if __name__ == "__main__":
    run_week4_optimization_demo()

