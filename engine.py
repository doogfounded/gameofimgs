from __future__ import annotations

import numpy as np
from PIL import Image


class LifeEngine:
    """Vectorized Conway's Game of Life engine with parent-inherited RGB colors."""

    def __init__(self, width: int = 160, height: int = 120) -> None:
        self.width: int = width
        self.height: int = height
        self.toroidal: bool = True
        self.generation: int = 0

        # Current state
        self.alive: np.ndarray = np.zeros((height, width), dtype=bool)
        self.colors: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)

        # Generation 0 snapshot (from image quantization)
        self.initial_alive: np.ndarray = np.zeros((height, width), dtype=bool)
        self.initial_colors: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
        # Full-grid original image RGB so manually drawn cells can sample the image color
        self.reference_rgb: np.ndarray = np.full((height, width, 3), 255, dtype=np.uint8)

        # Stored source PIL image (RGBA) for re-quantizing when sliders change
        self.source_image: Image.Image | None = None

    @property
    def population(self) -> int:
        return int(np.count_nonzero(self.alive))

    @property
    def total_cells(self) -> int:
        return self.width * self.height

    def load_pil_image(
        self,
        image: Image.Image,
        max_dim: int = 200,
        threshold: int = 128,
        invert: bool = False,
    ) -> None:
        """Store a PIL image and quantize it onto the grid."""
        self.source_image = image.convert("RGBA")
        self.apply_image_settings(max_dim=max_dim, threshold=threshold, invert=invert)

    def apply_image_settings(
        self,
        max_dim: int = 200,
        threshold: int = 128,
        invert: bool = False,
    ) -> None:
        """Re-quantize the stored source_image using the given resolution and threshold."""
        if self.source_image is None:
            return

        orig_w, orig_h = self.source_image.size
        if orig_w <= 0 or orig_h <= 0:
            return

        # Calculate grid dimensions preserving aspect ratio
        if orig_w >= orig_h:
            new_w = max(10, int(max_dim))
            new_h = max(10, int(round(orig_h * (new_w / orig_w))))
        else:
            new_h = max(10, int(max_dim))
            new_w = max(10, int(round(orig_w * (new_h / orig_h))))

        resized = self.source_image.resize((new_w, new_h), Image.Resampling.BOX)
        rgba = np.asarray(resized, dtype=np.uint8)

        rgb = rgba[:, :, :3].copy()
        alpha = rgba[:, :, 3]

        # Perceptual luminance (ITU-R BT.601)
        luminance = (
            0.299 * rgb[:, :, 0].astype(np.float32)
            + 0.587 * rgb[:, :, 1].astype(np.float32)
            + 0.114 * rgb[:, :, 2].astype(np.float32)
        )

        if invert:
            alive_mask = luminance <= float(threshold)
        else:
            alive_mask = luminance >= float(threshold)

        # Transparent pixels (alpha < 128) are always dead
        alive_mask &= alpha >= 128

        self.width = new_w
        self.height = new_h
        self.reference_rgb = rgb

        # Boost very dark alive colors slightly (e.g. when invert=True on black line art)
        # so they remain visible against the dark canvas background while keeping hue.
        display_rgb = rgb.copy()
        if invert:
            # Ensure alive dark pixels have minimum visibility brightness in color mode
            lum_int = luminance.astype(np.uint8)
            too_dark = alive_mask & (lum_int < 45)
            if np.any(too_dark):
                # Map dark ink to a soft glowing tint preserving any subtle color channels
                display_rgb[too_dark] = np.clip(
                    display_rgb[too_dark].astype(np.int16) + 95, 95, 255
                ).astype(np.uint8)

        self.initial_alive = alive_mask.copy()
        self.initial_colors = np.zeros_like(display_rgb)
        self.initial_colors[alive_mask] = display_rgb[alive_mask]

        self.reset_to_initial()

    def reset_to_initial(self) -> None:
        """Reset the grid to Generation 0 of the quantized image."""
        self.alive = self.initial_alive.copy()
        self.colors = self.initial_colors.copy()
        self.generation = 0

    def clear(self) -> None:
        """Clear all cells on the current grid."""
        self.alive.fill(False)
        self.colors.fill(0)
        self.generation = 0

    def step(self) -> None:
        """Advance Conway's Game of Life by 1 generation with parent-color inheritance."""
        pad_mode = "wrap" if self.toroidal else "constant"

        # Pad alive array by 1 cell on all sides
        if self.toroidal:
            padded_alive = np.pad(self.alive.astype(np.uint8), 1, mode=pad_mode)
            padded_colors = np.pad(
                self.colors.astype(np.uint16), ((1, 1), (1, 1), (0, 0)), mode=pad_mode
            )
        else:
            padded_alive = np.pad(
                self.alive.astype(np.uint8), 1, mode="constant", constant_values=0
            )
            padded_colors = np.pad(
                self.colors.astype(np.uint16),
                ((1, 1), (1, 1), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        h, w = self.height, self.width

        # Sum 8 neighbors for alive counts
        neighbors = (
            padded_alive[0:h, 0:w]
            + padded_alive[0:h, 1 : w + 1]
            + padded_alive[0:h, 2 : w + 2]
            + padded_alive[1 : h + 1, 0:w]
            + padded_alive[1 : h + 1, 2 : w + 2]
            + padded_alive[2 : h + 2, 0:w]
            + padded_alive[2 : h + 2, 1 : w + 1]
            + padded_alive[2 : h + 2, 2 : w + 2]
        )

        survives = self.alive & ((neighbors == 2) | (neighbors == 3))
        born = (~self.alive) & (neighbors == 3)
        next_alive = survives | born

        # Sum 8 neighbor RGB colors (only alive cells have non-zero RGB in self.colors)
        color_sum = (
            padded_colors[0:h, 0:w]
            + padded_colors[0:h, 1 : w + 1]
            + padded_colors[0:h, 2 : w + 2]
            + padded_colors[1 : h + 1, 0:w]
            + padded_colors[1 : h + 1, 2 : w + 2]
            + padded_colors[2 : h + 2, 0:w]
            + padded_colors[2 : h + 2, 1 : w + 1]
            + padded_colors[2 : h + 2, 2 : w + 2]
        )

        next_colors = np.zeros_like(self.colors)
        # Surviving cells keep their exact color
        next_colors[survives] = self.colors[survives]
        # Newborn cells inherit the average color of their 3 parent neighbors
        if np.any(born):
            next_colors[born] = (color_sum[born] // 3).astype(np.uint8)

        self.alive = next_alive
        self.colors = next_colors
        self.generation += 1

    def paint_cells(
        self,
        cx: int,
        cy: int,
        draw_alive: bool,
        brush_radius: int = 0,
        fallback_color: tuple[int, int, int] = (0, 255, 160),
    ) -> bool:
        """Draw or erase cells around (cx, cy). Returns True if any cell changed."""
        if cx < 0 or cx >= self.width or cy < 0 or cy >= self.height:
            return False

        y0 = max(0, cy - brush_radius)
        y1 = min(self.height, cy + brush_radius + 1)
        x0 = max(0, cx - brush_radius)
        x1 = min(self.width, cx + brush_radius + 1)

        if draw_alive:
            sub_alive = self.alive[y0:y1, x0:x1]
            newly_alive = ~sub_alive
            if not np.any(newly_alive):
                return False
            self.alive[y0:y1, x0:x1] = True

            # Use original image pixel color where reasonably bright, else fallback_color
            ref_patch = self.reference_rgb[y0:y1, x0:x1]
            ref_lum = (
                0.299 * ref_patch[:, :, 0]
                + 0.587 * ref_patch[:, :, 1]
                + 0.114 * ref_patch[:, :, 2]
            )
            patch_colors = self.colors[y0:y1, x0:x1]
            use_ref = newly_alive & (ref_lum >= 40)
            use_fallback = newly_alive & (ref_lum < 40)
            patch_colors[use_ref] = ref_patch[use_ref]
            patch_colors[use_fallback] = np.array(fallback_color, dtype=np.uint8)
            return True
        else:
            if not np.any(self.alive[y0:y1, x0:x1]):
                return False
            self.alive[y0:y1, x0:x1] = False
            self.colors[y0:y1, x0:x1] = 0
            return True

    def render_rgb_frame(
        self,
        monochrome: bool = False,
        mono_fg: tuple[int, int, int] = (0, 230, 140),
        bg_color: tuple[int, int, int] = (18, 18, 24),
    ) -> np.ndarray:
        """Render the current grid state into an (H, W, 3) uint8 RGB buffer."""
        frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = bg_color
        if monochrome:
            frame[self.alive] = mono_fg
        else:
            frame[self.alive] = self.colors[self.alive]
        return frame


def create_sample_image(width: int = 320, height: int = 240) -> Image.Image:
    """Generate a vibrant procedural sample image (neon cyber-sun & geometric waves)."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = (x - width * 0.5) / (width * 0.5)
    ny = (y - height * 0.45) / (height * 0.45)
    dist = np.sqrt(nx**2 + ny**2)

    img = np.zeros((height, width, 4), dtype=np.uint8)
    img[:, :, 3] = 255  # Opaque

    # Synthwave sun with horizontal stripes
    sun_mask = (dist < 0.62) & (np.sin(y * 0.35) > -0.35 * (ny + 0.3))
    img[sun_mask, 0] = 255
    img[sun_mask, 1] = np.clip(220 - ny[sun_mask] * 180, 40, 240).astype(np.uint8)
    img[sun_mask, 2] = np.clip(80 + ny[sun_mask] * 160, 60, 255).astype(np.uint8)

    # Concentric plasma rings outside the sun
    rings = (dist >= 0.68) & (dist < 1.15) & (np.cos(dist * 28.0) > 0.25)
    img[rings, 0] = np.clip(60 + (nx[rings] + 1) * 90, 40, 255).astype(np.uint8)
    img[rings, 1] = 230
    img[rings, 2] = 255

    # Perspective horizon grid at bottom
    horizon = y > height * 0.72
    hy = (y - height * 0.72) / (height * 0.28 + 1e-5)
    grid_pattern = (np.abs(np.sin(nx / (hy + 0.15) * 8.0)) > 0.82) | (
        np.abs(np.sin(hy * 18.0)) > 0.78
    )
    wave_mask = horizon & grid_pattern
    img[wave_mask, 0] = 180
    img[wave_mask, 1] = 70
    img[wave_mask, 2] = 255

    return Image.fromarray(img, mode="RGBA")

