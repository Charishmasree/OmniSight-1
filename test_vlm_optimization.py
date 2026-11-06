"""
OmniSight - Member 2: AI Integration using Vision-Language Models (VLM)
Week 4: Unit Tests for VLM Optimization (tests/test_vlm_optimization.py)

Comprehensive test suite verifying:
1. Image Chunking:
   - Slicing tall/large webpage screenshots into overlapping sections
   - Overlap margin calculation and boundary preservation
   - Detection and skipping of uniform blank/whitespace chunks
   - Result merging and deduplication across chunk boundaries
2. Image Cropping:
   - Safe bounding box cropping with boundary clamping
   - Focused ROI extraction from issue metadata
   - Automated detection of suspicious UI components (overlapping text, clipped elements, misaligned buttons)
3. VLM Optimization Pipeline:
   - Full analyze_optimized pipeline in offline mock mode (zero API key cost)
   - Chunked analysis workflow (analyze_chunks)
   - Micro-region ROI analysis workflow (analyze_roi)
4. Token Reduction:
   - Accurate multimodal vision token estimation
   - Token and byte compression ratio calculation (>75% token reduction on cropped ROIs)
5. Latency Improvement:
   - Latency tracking across preprocessing and inference
   - Speedup metrics from skipping blank chunks
"""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw

from ai.optimizer import (
    ImageChunker,
    ImageCropper,
    VLMTokenOptimizer,
    PerformanceTracker,
    VLMOptimizer,
    ChunkMetadata,
    ROICandidate,
    TokenMetrics,
)
from ai.vision_analyzer import (
    VisionAnalyzer,
    analyze_screenshot,
    analyze_optimized,
)
from ai.prompts import (
    build_roi_analysis_prompt,
    build_chunk_analysis_prompt,
    get_mock_roi_response,
)
from config import ImageNotFoundError


class TestVLMOptimizationWeek4(unittest.TestCase):
    """Test suite for Week 4 VLM Optimization features."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

        # Create a standard 1200x800 test UI screenshot with intentional anomalies
        self.standard_img = Image.new("RGB", (1200, 800), color=(245, 247, 250))
        draw = ImageDraw.Draw(self.standard_img)
        # Navbar
        draw.rectangle([(0, 0), (1200, 70)], fill=(20, 24, 33))
        # Hero section
        draw.rectangle([(40, 110), (1160, 320)], fill=(255, 255, 255))
        # Subtitle overlapping CTA button
        draw.rectangle([(70, 230), (220, 280)], fill=(79, 70, 229))
        draw.text((80, 245), "CTA Button", fill=(255, 255, 255))
        draw.text((80, 225), "Colliding subtitle text", fill=(239, 68, 68))
        # Metrics cards
        for i in range(3):
            draw.rectangle([(40 + i * 380, 360), (380 + i * 380, 640)], fill=(255, 255, 255))

        self.standard_img_path = self.dir_path / "standard_ui.png"
        self.standard_img.save(self.standard_img_path)

        # Create a tall 1200x2400 screenshot for chunking tests with a blank bottom section
        self.tall_img = Image.new("RGB", (1200, 2400), color=(255, 255, 255))
        tall_draw = ImageDraw.Draw(self.tall_img)
        tall_draw.rectangle([(0, 0), (1200, 800)], fill=(240, 240, 240))
        tall_draw.rectangle([(100, 100), (400, 300)], fill=(79, 70, 229))  # Top content
        tall_draw.rectangle([(0, 800), (1200, 1600)], fill=(230, 230, 230))
        tall_draw.rectangle([(100, 900), (500, 1100)], fill=(16, 185, 129))  # Middle content
        # Bottom (1600-2400) remains flat white -> should be identified as blank!
        self.tall_img_path = self.dir_path / "tall_ui.png"
        self.tall_img.save(self.tall_img_path)

        # Sample HTML DOM
        self.sample_html = """
        <div class="hero">
            <h1 class="hero-title">OmniSight Platform</h1>
            <p class="hero-subtitle" style="margin-bottom: -20px;">Overlapping subtitle</p>
            <button class="cta-btn">Click Me</button>
        </div>
        <div class="cards">
            <div class="metric-card" style="overflow: hidden;"><h3 class="metric-card-title">Clipped Text</h3></div>
        </div>
        """

    def tearDown(self):
        self.temp_dir.cleanup()

    # =========================================================================
    # 1. Image Chunking Tests
    # =========================================================================

    def test_chunking_tall_image_creates_correct_slices(self):
        """Should slice a 1200x2400 screenshot into overlapping chunks."""
        chunker = ImageChunker(chunk_width=1200, chunk_height=800, overlap_px=60)
        chunks = chunker.chunk_image(self.tall_img, vertical_only=True)

        self.assertGreater(len(chunks), 1)
        # Verify first chunk dimensions and coordinates
        self.assertEqual(chunks[0].bbox[0], 0)
        self.assertEqual(chunks[0].bbox[1], 0)
        self.assertEqual(chunks[0].bbox[2], 1200)
        self.assertEqual(chunks[0].bbox[3], 800)

        # Verify second chunk includes overlap
        self.assertLess(chunks[1].bbox[1], 800)  # Starts before 800 due to 60px overlap
        self.assertEqual(chunks[1].bbox[1], 740)

        # Verify all chunks cover the full height
        last_chunk = chunks[-1]
        self.assertEqual(last_chunk.bbox[3], 2400)

    def test_blank_chunk_detection(self):
        """Should detect flat/uniform whitespace chunks and flag them as is_blank."""
        chunker = ImageChunker(chunk_width=1200, chunk_height=800, blank_stddev_threshold=3.0)

        # Flat white image -> blank
        flat_img = Image.new("RGB", (400, 300), color=(255, 255, 255))
        is_blank, reason = chunker.is_blank_chunk(flat_img)
        self.assertTrue(is_blank)
        self.assertIn("Uniform background", reason)

        # High content image -> not blank
        is_blank_content, _ = chunker.is_blank_chunk(self.standard_img)
        self.assertFalse(is_blank_content)

    def test_chunking_skips_blank_chunks_for_tall_page(self):
        """Should identify bottom flat whitespace as blank in a tall screenshot."""
        chunker = ImageChunker(chunk_width=1200, chunk_height=800, overlap_px=60)
        chunks = chunker.chunk_image(self.tall_img)

        blank_count = sum(1 for c in chunks if c.is_blank)
        self.assertGreaterEqual(blank_count, 1)

    def test_merge_chunk_results_deduplicates_and_aggregates(self):
        """Should merge issues across chunks without duplicating identical selectors."""
        chunker = ImageChunker()
        chunk1_meta = ChunkMetadata(
            chunk_id=0, bbox=(0, 0, 1200, 800), width=1200, height=800, row=0, col=0, image=self.standard_img
        )
        chunk2_meta = ChunkMetadata(
            chunk_id=1, bbox=(0, 740, 1200, 1540), width=1200, height=800, row=1, col=0, image=self.standard_img
        )

        res1 = {
            "visual_health_score": 75,
            "issues": [
                {
                    "category": "overlapping_text",
                    "selector": ".hero-subtitle",
                    "suggested_fix": {"css_code": ".hero-subtitle { margin-bottom: 24px; }"},
                }
            ],
            "ui_elements": {"buttons": [{"label": "CTA"}]},
        }
        # Duplicate issue reported in overlap region of chunk 2
        res2 = {
            "visual_health_score": 85,
            "issues": [
                {
                    "category": "overlapping_text",
                    "selector": ".hero-subtitle",
                    "suggested_fix": {"css_code": ".hero-subtitle { margin-bottom: 24px; }"},
                },
                {
                    "category": "clipped_elements",
                    "selector": ".metric-card-title",
                    "suggested_fix": {"css_code": ".metric-card-title { text-overflow: ellipsis; }"},
                },
            ],
            "ui_elements": {"buttons": [{"label": "Export"}]},
        }

        merged = chunker.merge_chunk_results([res1, res2], [chunk1_meta, chunk2_meta])

        self.assertEqual(len(merged["issues"]), 2)  # Deduplicated from 3 to 2
        self.assertEqual(merged["visual_health_score"], 80)  # (75 + 85) / 2
        self.assertIn(".hero-subtitle", merged["css_fixes"])
        self.assertIn(".metric-card-title", merged["css_fixes"])
        self.assertEqual(len(merged["ui_elements"]["buttons"]), 2)

    # =========================================================================
    # 2. Image Cropping & Suspicious Region Detection Tests
    # =========================================================================

    def test_crop_region_with_padding_and_bounds_clamping(self):
        """Should crop given bounding box with padding without exceeding image boundaries."""
        cropper = ImageCropper()
        # Crop near upper-left border with 50px padding
        crop = cropper.crop_region(self.standard_img, bbox=(10, 10, 100, 100), padding=50)

        # Clamped to 0 at top-left: left=0, top=0, right=150, bottom=150
        self.assertEqual(crop.size, (150, 150))

        # Crop near bottom-right border
        crop_br = cropper.crop_region(self.standard_img, bbox=(1150, 750, 1195, 795), padding=50)
        self.assertLessEqual(crop_br.width, 100)
        self.assertLessEqual(crop_br.height, 100)

    def test_crop_for_issue_matches_presets(self):
        """Should locate and crop relevant region based on issue description/selector."""
        issue = {
            "title": "Hero subtitle collides with CTA button",
            "selector": ".hero-subtitle",
            "location": "Hero banner center",
        }
        cropped_img, bbox = ImageCropper.crop_for_issue(self.standard_img, issue, padding=20)

        self.assertIsNotNone(cropped_img)
        self.assertGreater(cropped_img.width, 0)
        self.assertGreater(cropped_img.height, 0)
        self.assertEqual(len(bbox), 4)

    def test_detect_suspicious_regions_finds_anomalies(self):
        """Should detect overlapping text and clipped element candidates using heuristics."""
        candidates = ImageCropper.detect_suspicious_regions(
            self.standard_img, html_content=self.sample_html, max_regions=3
        )

        self.assertGreater(len(candidates), 0)
        labels = [c.label for c in candidates]
        self.assertIn("overlapping_text", labels)

        # Verify candidate metadata structure
        cand = candidates[0]
        self.assertTrue(cand.roi_id.startswith("ROI-"))
        self.assertGreaterEqual(cand.confidence, 0.70)
        self.assertIsNotNone(cand.cropped_image)
        self.assertEqual(cand.cropped_image.size[0], cand.bbox[2] - cand.bbox[0] + 50)  # with padding

    # =========================================================================
    # 3. Token Reduction & Optimization Tests
    # =========================================================================

    def test_token_estimation_formula(self):
        """Should calculate expected Gemini multimodal token costs."""
        # 400x200 fits in single 768x768 tile -> 258 tokens
        self.assertEqual(VLMTokenOptimizer.estimate_tokens(400, 200), 258)
        # 1200x800 requires ceil(1200/768)=2 * ceil(800/768)=2 = 4 tiles -> 1,032 tokens
        self.assertEqual(VLMTokenOptimizer.estimate_tokens(1200, 800), 1032)

    def test_token_reduction_on_cropped_roi(self):
        """Cropping to a focused component should yield >= 70% token reduction vs full page."""
        optimizer = VLMTokenOptimizer()
        full_tokens = optimizer.estimate_tokens(1200, 800)

        # Crop a 300x150 badge
        crop_img = self.standard_img.crop((70, 220, 370, 370))
        _, _, metrics = optimizer.optimize_image(crop_img, baseline_dimensions=(1200, 800))

        self.assertGreaterEqual(metrics.token_reduction_pct, 70.0)
        self.assertGreater(metrics.estimated_tokens_saved, 500)
        self.assertGreater(metrics.pixel_reduction_pct, 85.0)

    def test_image_payload_compression_rgba_to_rgb(self):
        """Should convert RGBA images cleanly to JPEG format with white background."""
        rgba_img = Image.new("RGBA", (200, 200), color=(100, 150, 200, 180))
        opt_bytes, mime, metrics = VLMTokenOptimizer.optimize_image(rgba_img, target_format="JPEG")

        self.assertEqual(mime, "image/jpeg")
        self.assertGreater(len(opt_bytes), 0)
        # Check that it loads back as valid JPEG
        reloaded = Image.open(io.BytesIO(opt_bytes))
        self.assertEqual(reloaded.format, "JPEG")

    # =========================================================================
    # 4. Latency & Performance Benchmark Tracker Tests
    # =========================================================================

    def test_performance_tracker_records_metrics(self):
        """Should record preprocessing and inference timestamps and compute speedup."""
        tracker = PerformanceTracker()
        tracker.start()
        tracker.stop_preprocessing()
        tracker.stop_inference()
        metrics = tracker.get_metrics(chunks_total=4, chunks_skipped_blank=2, chunks_analyzed=2)

        self.assertGreaterEqual(metrics.total_time_ms, 0.0)
        self.assertEqual(metrics.chunks_total, 4)
        self.assertEqual(metrics.chunks_skipped_blank, 2)
        self.assertEqual(metrics.chunks_analyzed, 2)
        self.assertEqual(metrics.speedup_factor_vs_full, 2.0)  # 4 / 2 = 2.0x speedup

    # =========================================================================
    # 5. Full Pipeline Integration Tests (Mock Mode - Zero Cost)
    # =========================================================================

    def test_analyze_optimized_full_pipeline_mock_mode(self):
        """Should run analyze_optimized end-to-end and populate all optimization metadata."""
        analyzer = VisionAnalyzer(mock_mode=True)
        report = analyzer.analyze_optimized(
            screenshot=self.standard_img_path,
            html_content=self.sample_html,
            page_url="https://omnisight.local/dashboard",
        )

        self.assertEqual(report["status"], "success")
        self.assertIn("optimization", report)
        opt = report["optimization"]

        # Verify structured optimization blocks
        self.assertIn("token_metrics", opt)
        self.assertIn("performance_metrics", opt)
        self.assertIn("detected_rois", opt)

        # Verify token reduction recorded
        self.assertIn("estimated_tokens_saved", opt["token_metrics"])
        self.assertIn("token_reduction_pct", opt["token_metrics"])

        # Verify metadata flags
        meta = report["metadata"]
        self.assertTrue(meta["optimization_enabled"])
        self.assertIn("tokens_saved_estimate", meta)

    def test_analyze_chunks_pipeline_mock_mode(self):
        """Should run chunked analysis pipeline across a tall screenshot."""
        analyzer = VisionAnalyzer(mock_mode=True)
        report = analyzer.analyze_chunks(
            screenshot=self.tall_img_path,
            chunk_width=1200,
            chunk_height=800,
            overlap_px=60,
            vertical_only=True,
            html_content=self.sample_html,
        )

        self.assertEqual(report["status"], "success")
        self.assertIn("optimization", report)
        chunk_opt = report["optimization"]["chunk_metrics"]
        self.assertGreater(chunk_opt["chunks_total"], 1)
        self.assertIn("chunks_skipped_blank", chunk_opt)

    def test_analyze_roi_focused_micro_region(self):
        """Should run focused micro-region ROI analysis."""
        analyzer = VisionAnalyzer(mock_mode=True)
        roi_bbox = (50, 150, 350, 300)
        result = analyzer.analyze_roi(
            screenshot=self.standard_img_path,
            roi_bbox=roi_bbox,
            roi_label="hero_cta_badge",
        )

        self.assertEqual(result["status"], "success")
        self.assertIn("optimization", result)
        self.assertEqual(result["optimization"]["roi_bbox"], list(roi_bbox))
        self.assertEqual(result["optimization"]["mode"], "roi_crop")

    def test_convenience_function_with_optimize_flag(self):
        """analyze_screenshot(..., optimize=True) should execute the optimization pipeline."""
        result = analyze_screenshot(
            screenshot=self.standard_img_path,
            mock_mode=True,
            optimize=True,
        )

        self.assertEqual(result["status"], "success")
        self.assertIn("optimization", result)


if __name__ == "__main__":
    unittest.main()
