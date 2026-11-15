"""fastblur/render_v4.py — FAST v4 Production Blur Renderer

Rendering improvements over v3 (anonymize.py):
  - Faces: axis-aligned rotated ellipse mask inscribed in the padded bounding box.
    Cleaner shape that follows the natural oval of a head, eliminating corner artifacts
    and preventing unnecessary blurring of hair and background.
  - Edge blending / anti-aliasing: 1–3 px Gaussian feather on mask boundary.
    Creates studio-grade seamless alpha transition between blurred region and raw image.
  - Plates: padded rectangular box with the same feathered edge blending.
  - Pixel invariant: bit-identical pixels outside the feathered transition zone.
  - Tiny-box fallback: boxes with min side < min_ellipse_side_px use rectangular blur.
  - Deterministic: bit-identical output across runs for same input and config.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


@dataclass
class RenderConfig:
    face_mask: Literal["ellipse", "rect"] = "ellipse"
    plate_mask: Literal["feathered_rect", "rect"] = "feathered_rect"
    feather_sigma_px: float = 2.0
    face_pad_ratio: float = 0.05
    plate_pad_ratio: float = 0.10
    min_ellipse_side_px: int = 8
    bit_identical_outside_mask: bool = True
    # Blur settings
    blur_method: Literal["adaptive", "fixed"] = "adaptive"
    fixed_kernel: int = 51
    adaptive_target_px: int = 24
    min_sigma: float = 5.0

    @classmethod
    def from_dict(cls, d: dict) -> "RenderConfig":
        render_cfg = d.get("render", {}) if isinstance(d, dict) else {}
        blur_cfg = d.get("blur", {}) if isinstance(d, dict) else {}
        return cls(
            face_mask=render_cfg.get("face_mask", "ellipse"),
            plate_mask=render_cfg.get("plate_mask", "feathered_rect"),
            feather_sigma_px=float(render_cfg.get("feather_sigma_px", 2.0)),
            face_pad_ratio=float(render_cfg.get("face_pad_ratio", 0.05)),
            plate_pad_ratio=float(render_cfg.get("plate_pad_ratio", 0.10)),
            min_ellipse_side_px=int(render_cfg.get("min_ellipse_side_px", 8)),
            bit_identical_outside_mask=bool(render_cfg.get("bit_identical_outside_mask", True)),
            blur_method=blur_cfg.get("method", "adaptive"),
            fixed_kernel=int(blur_cfg.get("fixed_kernel", 51)),
            adaptive_target_px=int(blur_cfg.get("adaptive_target_px", 24)),
            min_sigma=float(blur_cfg.get("min_sigma", 5.0)),
        )


@dataclass
class BlurStats:
    faces_rendered: int = 0
    plates_rendered: int = 0
    blurred_area_px: int = 0
    face_mask_type: str = "ellipse"
    plate_mask_type: str = "feathered_rect"
    blur_method: str = "adaptive"


def _create_feathered_mask(
    roi_h: int,
    roi_w: int,
    mask_type: str,
    feather_sigma: float,
    center: tuple[int, int] | None = None,
    axes: tuple[int, int] | None = None,
) -> np.ndarray:
    """Create a 2D float32 mask in [0.0, 1.0] with feathered boundary."""
    if roi_h <= 0 or roi_w <= 0:
        return np.zeros((max(0, roi_h), max(0, roi_w)), dtype=np.float32)

    # 1. Binary base mask (uint8: 0 or 255)
    binary_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)

    if mask_type == "ellipse" and center is not None and axes is not None:
        ax_w, ax_h = max(1, axes[0]), max(1, axes[1])
        cv2.ellipse(
            binary_mask,
            center=center,
            axes=(ax_w, ax_h),
            angle=0.0,
            startAngle=0.0,
            endAngle=360.0,
            color=255,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    else:
        # Rectangular mask
        binary_mask.fill(255)

    # 2. Feathering via Gaussian blur on the mask
    if feather_sigma <= 0.05:
        return (binary_mask > 127).astype(np.float32)

    # Kernel size covers ~3 sigma in each direction, odd integer
    ksize = int(math.ceil(feather_sigma * 6.0)) | 1
    ksize = max(3, min(ksize, 31))

    blurred_mask = cv2.GaussianBlur(
        binary_mask.astype(np.float32) / 255.0,
        (ksize, ksize),
        sigmaX=feather_sigma,
        sigmaY=feather_sigma,
        borderType=cv2.BORDER_CONSTANT,
    )
    return np.clip(blurred_mask, 0.0, 1.0)


def _compute_blurred_roi(roi: np.ndarray, cfg: RenderConfig) -> np.ndarray:
    """Blur an ROI using adaptive downscale-blur or fixed Gaussian kernel."""
    h, w = roi.shape[:2]
    if h < 2 or w < 2:
        return roi.copy()

    if cfg.blur_method == "adaptive":
        short_side = min(h, w)
        if short_side <= cfg.adaptive_target_px:
            # ROI is already tiny; strong Gaussian blur directly
            k = max(3, (short_side // 2) * 2 + 1)
            return cv2.GaussianBlur(roi, (k, k), sigmaX=max(cfg.min_sigma, short_side / 4.0))

        scale = cfg.adaptive_target_px / float(short_side)
        small_w = max(2, int(round(w * scale)))
        small_h = max(2, int(round(h * scale)))

        # Downscale
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)

        # Gaussian blur at reduced scale
        k_small = 5
        blurred_small = cv2.GaussianBlur(small, (k_small, k_small), sigmaX=cfg.min_sigma / 2.0)

        # Upscale back to original ROI size
        return cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Fixed method
    k = max(3, cfg.fixed_kernel | 1)
    sigma = max(cfg.min_sigma, k / 6.0)
    return cv2.GaussianBlur(roi, (k, k), sigmaX=sigma)


def _apply_blended_blur(
    img: np.ndarray,
    roi_coords: tuple[int, int, int, int],  # x1, y1, x2, y2
    mask: np.ndarray,
    cfg: RenderConfig,
) -> int:
    """Alpha blend blurred ROI into image in-place. Returns number of blurred pixels."""
    x1, y1, x2, y2 = roi_coords
    roi_h, roi_w = y2 - y1, x2 - x1
    if roi_h <= 0 or roi_w <= 0:
        return 0

    roi = img[y1:y2, x1:x2]
    blurred_roi = _compute_blurred_roi(roi, cfg)

    if cfg.feather_sigma_px <= 0.05:
        # Binary mask fast path
        bin_mask = mask > 0.5
        roi[bin_mask] = blurred_roi[bin_mask]
        return int(np.sum(bin_mask))

    # Alpha blending with bit-identical invariant outside transition
    alpha = mask[:, :, np.newaxis]  # (H, W, 1) float32

    # Pixels with alpha > 0 get blended
    active = mask > 1e-4
    if not np.any(active):
        return 0

    if cfg.bit_identical_outside_mask:
        # Blend only where alpha > 0, keep bit-identical elsewhere
        fg = blurred_roi.astype(np.float32)
        bg = roi.astype(np.float32)
        blended = fg * alpha + bg * (1.0 - alpha)
        blended_uint8 = np.clip(np.round(blended), 0, 255).astype(np.uint8)

        # Write back only active pixels
        roi[active] = blended_uint8[active]
    else:
        fg = blurred_roi.astype(np.float32)
        bg = roi.astype(np.float32)
        blended = fg * alpha + bg * (1.0 - alpha)
        img[y1:y2, x1:x2] = np.clip(np.round(blended), 0, 255).astype(np.uint8)

    return int(np.sum(mask > 0.5))


def render_face_blur(img: np.ndarray, box: np.ndarray, cfg: RenderConfig) -> int:
    """Render face blur on img in-place. box is [x1, y1, x2, y2, ...] float."""
    h_img, w_img = img.shape[:2]
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return 0

    # Expand box slightly to cover jawline and temples
    pad_w = bw * cfg.face_pad_ratio
    pad_h = bh * cfg.face_pad_ratio
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5

    x1_pad = max(0, int(math.floor(cx - (bw * 0.5 + pad_w))))
    y1_pad = max(0, int(math.floor(cy - (bh * 0.5 + pad_h))))
    x2_pad = min(w_img, int(math.ceil(cx + (bw * 0.5 + pad_w))))
    y2_pad = min(h_img, int(math.ceil(cy + (bh * 0.5 + pad_h))))

    roi_w = x2_pad - x1_pad
    roi_h = y2_pad - y1_pad
    if roi_w <= 2 or roi_h <= 2:
        return 0

    min_side = min(roi_w, roi_h)
    use_ellipse = (cfg.face_mask == "ellipse") and (min_side >= cfg.min_ellipse_side_px)

    if use_ellipse:
        # Ellipse center in ROI coordinates
        c_roi_x = int(round(cx - x1_pad))
        c_roi_y = int(round(cy - y1_pad))
        ax_w = max(1, int(round((bw * 0.5 + pad_w))))
        ax_h = max(1, int(round((bh * 0.5 + pad_h))))
        mask = _create_feathered_mask(
            roi_h, roi_w,
            mask_type="ellipse",
            feather_sigma=cfg.feather_sigma_px,
            center=(c_roi_x, c_roi_y),
            axes=(ax_w, ax_h),
        )
    else:
        # Rectangular fallback
        mask = _create_feathered_mask(
            roi_h, roi_w,
            mask_type="rect",
            feather_sigma=cfg.feather_sigma_px if cfg.face_mask != "rect" else 0.0,
        )

    return _apply_blended_blur(img, (x1_pad, y1_pad, x2_pad, y2_pad), mask, cfg)


def render_plate_blur(img: np.ndarray, box: np.ndarray, cfg: RenderConfig) -> int:
    """Render license plate blur on img in-place."""
    h_img, w_img = img.shape[:2]
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return 0

    # Expand box by plate_pad_ratio
    pad_w = bw * cfg.plate_pad_ratio
    pad_h = bh * cfg.plate_pad_ratio
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5

    x1_pad = max(0, int(math.floor(cx - (bw * 0.5 + pad_w))))
    y1_pad = max(0, int(math.floor(cy - (bh * 0.5 + pad_h))))
    x2_pad = min(w_img, int(math.ceil(cx + (bw * 0.5 + pad_w))))
    y2_pad = min(h_img, int(math.ceil(cy + (bh * 0.5 + pad_h))))

    roi_w = x2_pad - x1_pad
    roi_h = y2_pad - y1_pad
    if roi_w <= 2 or roi_h <= 2:
        return 0

    feather = cfg.feather_sigma_px if cfg.plate_mask == "feathered_rect" else 0.0
    mask = _create_feathered_mask(
        roi_h, roi_w,
        mask_type="rect",
        feather_sigma=feather,
    )
    return _apply_blended_blur(img, (x1_pad, y1_pad, x2_pad, y2_pad), mask, cfg)


def render_detections(
    img: np.ndarray,
    faces: np.ndarray,
    plates: np.ndarray,
    vehicles: np.ndarray | None,
    cfg: RenderConfig,
) -> BlurStats:
    """Main rendering entry for FAST v4. Replaces v3 anonymize().
    
    Modifies img in-place and returns BlurStats.
    """
    stats = BlurStats(
        face_mask_type=cfg.face_mask,
        plate_mask_type=cfg.plate_mask,
        blur_method=cfg.blur_method,
    )

    # Process faces
    if faces is not None and len(faces) > 0:
        for face in faces:
            px = render_face_blur(img, face, cfg)
            stats.faces_rendered += 1
            stats.blurred_area_px += px

    # Process plates
    if plates is not None and len(plates) > 0:
        for plate in plates:
            px = render_plate_blur(img, plate, cfg)
            stats.plates_rendered += 1
            stats.blurred_area_px += px

    return stats
