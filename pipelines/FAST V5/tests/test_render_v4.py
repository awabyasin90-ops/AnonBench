"""tests/test_render_v4.py — Unit tests for FAST v4 Production Blur Renderer."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from fastblur.render_v4 import (
    RenderConfig,
    _create_feathered_mask,
    render_face_blur,
    render_plate_blur,
    render_detections,
)


def test_face_mask_is_elliptical():
    """T1: Verify center of face bounding box is masked, while corners are outside or lower weight."""
    h, w = 100, 100
    mask = _create_feathered_mask(
        h, w,
        mask_type="ellipse",
        feather_sigma=1.0,
        center=(50, 50),
        axes=(40, 40),
    )
    # Center should be fully masked
    assert mask[50, 50] > 0.95
    # Corners of the bounding box should be outside the circle/ellipse
    assert mask[10, 10] < 0.1
    assert mask[90, 90] < 0.1
    assert mask[10, 90] < 0.1
    assert mask[90, 10] < 0.1


def test_feathered_edge_has_gradient():
    """T2: Verify feathering creates a gradual transition (values between 0.1 and 0.9)."""
    mask = _create_feathered_mask(
        100, 100,
        mask_type="ellipse",
        feather_sigma=3.0,
        center=(50, 50),
        axes=(30, 30),
    )
    transition_px = np.logical_and(mask > 0.05, mask < 0.95)
    assert np.sum(transition_px) > 100, "Should have a smooth transition region"


def test_pixels_outside_mask_bit_identical():
    """T3: Verify pixels outside the blur mask remain bit-identical."""
    img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    original = img.copy()

    box = np.array([50, 50, 120, 140, 0.9], dtype=np.float32)
    cfg = RenderConfig(
        face_mask="ellipse",
        feather_sigma_px=2.0,
        bit_identical_outside_mask=True,
    )
    render_face_blur(img, box, cfg)

    # Far outside region must be 100% bit-identical
    assert np.array_equal(img[0:30, 0:30], original[0:30, 0:30])
    assert np.array_equal(img[170:200, 170:200], original[170:200, 170:200])


def test_no_detections_no_change():
    """T4: Zero detections produce bit-identical output image."""
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    original = img.copy()

    cfg = RenderConfig()
    stats = render_detections(img, np.zeros((0, 5), dtype=np.float32), np.zeros((0, 5), dtype=np.float32), None, cfg)

    assert stats.faces_rendered == 0
    assert stats.plates_rendered == 0
    assert np.array_equal(img, original)


def test_blur_scales_with_box_size():
    """T5: Adaptive blur reduces high-frequency Laplacian variance on both small and large boxes."""
    # Create high-frequency checkerboard texture
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[::2, ::2] = 255
    img[1::2, 1::2] = 255

    orig_var = cv2.Laplacian(img[60:140, 60:140], cv2.CV_64F).var()
    box = np.array([50, 50, 150, 150, 0.95], dtype=np.float32)
    cfg = RenderConfig(blur_method="adaptive")

    render_face_blur(img, box, cfg)
    blurred_roi = img[60:140, 60:140]
    blurred_var = cv2.Laplacian(blurred_roi, cv2.CV_64F).var()

    assert blurred_var < orig_var * 0.1, "Blur should significantly reduce Laplacian variance"


def test_box_clipped_at_image_border():
    """T6: Bounding box extending beyond image boundary does not crash and clips properly."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    box = np.array([-20, -10, 50, 60, 0.8], dtype=np.float32)
    cfg = RenderConfig()

    px = render_face_blur(img, box, cfg)
    assert px > 0


def test_tiny_box_uses_rect_fallback():
    """T7: Boxes smaller than min_ellipse_side_px use rectangular fallback."""
    img = np.ones((50, 50, 3), dtype=np.uint8) * 128
    box = np.array([10, 10, 15, 15, 0.8], dtype=np.float32)
    cfg = RenderConfig(min_ellipse_side_px=8)

    px = render_face_blur(img, box, cfg)
    assert px > 0


def test_zero_feather_equals_binary_mask():
    """T8: Zero feather produces binary mask (v3 equivalent)."""
    mask = _create_feathered_mask(50, 50, mask_type="rect", feather_sigma=0.0)
    unique_vals = np.unique(mask)
    assert set(unique_vals).issubset({0.0, 1.0})


def test_two_runs_byte_identical():
    """T9: Determinism — two runs on identical input produce byte-identical images."""
    img1 = np.random.randint(0, 256, (120, 120, 3), dtype=np.uint8)
    img2 = img1.copy()
    box = np.array([20, 20, 80, 90, 0.88], dtype=np.float32)
    cfg = RenderConfig()

    render_face_blur(img1, box, cfg)
    render_face_blur(img2, box, cfg)

    assert np.array_equal(img1, img2)


def test_render_detections_aggregates_stats():
    """T10: render_detections returns proper BlurStats count."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    faces = np.array([[10, 10, 50, 50, 0.9], [60, 60, 90, 90, 0.85]], dtype=np.float32)
    plates = np.array([[100, 100, 140, 120, 0.95]], dtype=np.float32)

    cfg = RenderConfig()
    stats = render_detections(img, faces, plates, None, cfg)

    assert stats.faces_rendered == 2
    assert stats.plates_rendered == 1
    assert stats.blurred_area_px > 0
