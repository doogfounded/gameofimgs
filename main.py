from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, QTimer, QBuffer, QIODevice
from PyQt6.QtGui import (
    QAction,
    QDragEnterEvent,
    QDropEvent,
    QGuiApplication,
    QImage,
    QKeyEvent,
    QKeySequence,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from canvas import GridCanvas
from engine import LifeEngine, create_sample_image


DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #14151b;
    color: #e4e7ef;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}
QGroupBox {
    background-color: #1c1e27;
    border: 1px solid #2b2f3e;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 12px;
    font-weight: 600;
    color: #9aa5ce;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QPushButton {
    background-color: #282c3a;
    border: 1px solid #383e52;
    border-radius: 6px;
    padding: 7px 12px;
    color: #e4e7ef;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #32374a;
    border-color: #4f5873;
}
QPushButton:pressed {
    background-color: #202330;
}
QPushButton#primaryBtn {
    background-color: #00b876;
    border-color: #00d68a;
    color: #0a1410;
    font-weight: 700;
}
QPushButton#primaryBtn:hover {
    background-color: #00ce84;
}
QPushButton#recordBtnActive {
    background-color: #e03c52;
    border-color: #ff5c72;
    color: #ffffff;
    font-weight: 700;
}
QSlider::groove:horizontal {
    border: 1px solid #2e3344;
    height: 6px;
    background: #14151b;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #00c480;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #e4e7ef;
    border: 2px solid #00c480;
    width: 14px;
    margin: -5px 0;
    border-radius: 8px;
}
QComboBox {
    background-color: #242836;
    border: 1px solid #383e52;
    border-radius: 6px;
    padding: 5px 10px;
}
QComboBox::drop-down {
    border: none;
}
QCheckBox {
    spacing: 8px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #48506a;
    background: #14151b;
}
QCheckBox::indicator:checked {
    background: #00c480;
    border-color: #00e092;
}
QLabel#statValue {
    font-size: 16px;
    font-weight: 700;
    color: #00e599;
}
QLabel#subtleHint {
    color: #7c84a3;
    font-size: 11px;
}
"""


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Image to Conway's Game of Life")
        self.resize(1280, 820)
        self.setAcceptDrops(True)

        self.engine = LifeEngine()
        self.is_playing: bool = False
        self.is_recording_gif: bool = False
        self.recorded_frames: list[Image.Image] = []
        self.current_image_name: str = "Built-in Cyber-Sun Sample"

        # Simulation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_tick)

        self._build_ui()
        self._setup_shortcuts()

        # Load initial sample image
        sample = create_sample_image(320, 240)
        self._load_pil_image(sample, "Built-in Cyber-Sun Sample", fit_view=True)

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Left control sidebar inside a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(350)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        sidebar = QWidget()
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(14, 14, 14, 14)
        side_layout.setSpacing(10)

        # 1. Image Source Group
        img_group = QGroupBox("1. Image Source")
        img_layout = QVBoxLayout(img_group)

        btn_row1 = QHBoxLayout()
        self.open_btn = QPushButton("Open Image...")
        self.open_btn.clicked.connect(self.open_image_dialog)
        self.paste_btn = QPushButton("Paste (Ctrl+V)")
        self.paste_btn.clicked.connect(self.paste_image_from_clipboard)
        btn_row1.addWidget(self.open_btn)
        btn_row1.addWidget(self.paste_btn)
        img_layout.addLayout(btn_row1)

        self.sample_btn = QPushButton("Load Sample Artwork")
        self.sample_btn.clicked.connect(self.load_sample_artwork)
        img_layout.addWidget(self.sample_btn)

        self.source_label = QLabel(f"Loaded: {self.current_image_name}")
        self.source_label.setWordWrap(True)
        self.source_label.setObjectName("subtleHint")
        img_layout.addWidget(self.source_label)

        drop_hint = QLabel("Tip: Drag & drop any image file onto the window.")
        drop_hint.setObjectName("subtleHint")
        img_layout.addWidget(drop_hint)
        side_layout.addWidget(img_group)

        # 2. Image-to-Grid Quantization Group
        quant_group = QGroupBox("2. Grid & Brightness Threshold")
        quant_layout = QVBoxLayout(quant_group)

        res_header = QHBoxLayout()
        res_header.addWidget(QLabel("Grid Resolution (Longest Side):"))
        self.res_val_label = QLabel("200 px")
        self.res_val_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        res_header.addWidget(self.res_val_label)
        quant_layout.addLayout(res_header)

        self.res_slider = QSlider(Qt.Orientation.Horizontal)
        self.res_slider.setRange(50, 500)
        self.res_slider.setValue(200)
        self.res_slider.valueChanged.connect(self._on_quantization_changed)
        quant_layout.addWidget(self.res_slider)

        thresh_header = QHBoxLayout()
        thresh_header.addWidget(QLabel("Brightness Threshold:"))
        self.thresh_val_label = QLabel("115 / 255")
        self.thresh_val_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        thresh_header.addWidget(self.thresh_val_label)
        quant_layout.addLayout(thresh_header)

        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(0, 255)
        self.thresh_slider.setValue(115)
        self.thresh_slider.valueChanged.connect(self._on_quantization_changed)
        quant_layout.addWidget(self.thresh_slider)

        self.invert_chk = QCheckBox("Invert Threshold (Dark Pixels = Alive)")
        self.invert_chk.toggled.connect(self._on_quantization_changed)
        quant_layout.addWidget(self.invert_chk)

        side_layout.addWidget(quant_group)

        # 3. Simulation Playback Group
        sim_group = QGroupBox("3. Simulation Playback")
        sim_layout = QVBoxLayout(sim_group)

        transport_grid = QGridLayout()
        self.play_btn = QPushButton("▶  Play (P)")
        self.play_btn.setObjectName("primaryBtn")
        self.play_btn.clicked.connect(self.toggle_play)

        self.step_btn = QPushButton("⏭  Step (S)")
        self.step_btn.clicked.connect(self.step_once)

        self.reset_btn = QPushButton("↺  Reset to Image (R)")
        self.reset_btn.clicked.connect(self.reset_to_image)

        self.clear_btn = QPushButton("✕  Clear Grid (C)")
        self.clear_btn.clicked.connect(self.clear_grid)

        transport_grid.addWidget(self.play_btn, 0, 0)
        transport_grid.addWidget(self.step_btn, 0, 1)
        transport_grid.addWidget(self.reset_btn, 1, 0)
        transport_grid.addWidget(self.clear_btn, 1, 1)
        sim_layout.addLayout(transport_grid)

        fps_header = QHBoxLayout()
        fps_header.addWidget(QLabel("Simulation Speed:"))
        self.fps_val_label = QLabel("20 FPS")
        self.fps_val_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        fps_header.addWidget(self.fps_val_label)
        sim_layout.addLayout(fps_header)

        self.fps_slider = QSlider(Qt.Orientation.Horizontal)
        self.fps_slider.setRange(1, 60)
        self.fps_slider.setValue(20)
        self.fps_slider.valueChanged.connect(self._on_fps_changed)
        sim_layout.addWidget(self.fps_slider)

        self.toroidal_chk = QCheckBox("Wrap Grid Edges (Toroidal)")
        self.toroidal_chk.setChecked(True)
        self.toroidal_chk.toggled.connect(self._on_toroidal_toggled)
        sim_layout.addWidget(self.toroidal_chk)

        side_layout.addWidget(sim_group)

        # 4. Display & Canvas Interaction Group
        disp_group = QGroupBox("4. Display & Brush")
        disp_layout = QVBoxLayout(disp_group)

        disp_layout.addWidget(QLabel("Color Mode:"))
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(
            [
                "Original Image Colors (Inherit)",
                "Classic Monochrome",
            ]
        )
        self.color_mode_combo.currentIndexChanged.connect(self._on_color_mode_changed)
        disp_layout.addWidget(self.color_mode_combo)

        self.grid_lines_chk = QCheckBox("Show Grid Lines (when zoomed in)")
        self.grid_lines_chk.setChecked(True)
        self.grid_lines_chk.toggled.connect(self._on_grid_lines_toggled)
        disp_layout.addWidget(self.grid_lines_chk)

        brush_header = QHBoxLayout()
        brush_header.addWidget(QLabel("Draw/Erase Brush Size:"))
        self.brush_val_label = QLabel("1 × 1")
        self.brush_val_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        brush_header.addWidget(self.brush_val_label)
        disp_layout.addLayout(brush_header)

        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(0, 4)
        self.brush_slider.setValue(0)
        self.brush_slider.valueChanged.connect(self._on_brush_changed)
        disp_layout.addWidget(self.brush_slider)

        self.fit_btn = QPushButton("Fit Grid to Window (F)")
        self.fit_btn.clicked.connect(self.fit_canvas)
        disp_layout.addWidget(self.fit_btn)

        canvas_hint = QLabel(
            "Left-drag: Draw alive  |  Right-drag: Erase\n"
            "Scroll: Zoom at cursor  |  Middle/Space+drag: Pan"
        )
        canvas_hint.setObjectName("subtleHint")
        disp_layout.addWidget(canvas_hint)
        side_layout.addWidget(disp_group)

        # 5. Live Statistics Group
        stats_group = QGroupBox("5. Live Statistics")
        stats_grid = QGridLayout(stats_group)

        stats_grid.addWidget(QLabel("Generation:"), 0, 0)
        self.gen_label = QLabel("0")
        self.gen_label.setObjectName("statValue")
        stats_grid.addWidget(self.gen_label, 0, 1)

        stats_grid.addWidget(QLabel("Alive Cells:"), 1, 0)
        self.pop_label = QLabel("0 (0.0%)")
        self.pop_label.setObjectName("statValue")
        stats_grid.addWidget(self.pop_label, 1, 1)

        stats_grid.addWidget(QLabel("Grid Size:"), 2, 0)
        self.dim_label = QLabel("200 × 150")
        stats_grid.addWidget(self.dim_label, 2, 1)

        side_layout.addWidget(stats_group)

        # 6. Export Group
        export_group = QGroupBox("6. Export")
        export_layout = QVBoxLayout(export_group)

        self.save_png_btn = QPushButton("Save Current Frame (PNG)...")
        self.save_png_btn.clicked.connect(self.export_png)
        export_layout.addWidget(self.save_png_btn)

        gif_row = QHBoxLayout()
        self.rec_gif_btn = QPushButton("● Record GIF")
        self.rec_gif_btn.clicked.connect(self.toggle_record_gif)
        self.quick_gif_btn = QPushButton("Quick 100-Gen GIF...")
        self.quick_gif_btn.clicked.connect(self.quick_export_gif)
        gif_row.addWidget(self.rec_gif_btn)
        gif_row.addWidget(self.quick_gif_btn)
        export_layout.addLayout(gif_row)

        side_layout.addWidget(export_group)
        side_layout.addStretch(1)

        scroll.setWidget(sidebar)
        root_layout.addWidget(scroll)

        # Right side: Interactive Canvas
        self.canvas = GridCanvas(self.engine, self)
        self.canvas.cellsEdited.connect(self._update_stats_labels)
        self.canvas.imageDropped.connect(self.load_image_from_path)
        root_layout.addWidget(self.canvas, stretch=1)

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+V"), self, activated=self.paste_image_from_clipboard)
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.open_image_dialog)
        QShortcut(QKeySequence("P"), self, activated=self.toggle_play)
        QShortcut(QKeySequence("S"), self, activated=self.step_once)
        QShortcut(QKeySequence("R"), self, activated=self.reset_to_image)
        QShortcut(QKeySequence("C"), self, activated=self.clear_grid)
        QShortcut(QKeySequence("F"), self, activated=self.fit_canvas)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.canvas.set_space_pan(True)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.canvas.set_space_pan(False)
            event.accept()
            return
        super().keyReleaseEvent(event)

    # --- Image Loading ---

    def open_image_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select an Image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tiff);;All Files (*)",
        )
        if path:
            self.load_image_from_path(path)

    def load_image_from_path(self, path: str) -> None:
        try:
            pil_img = Image.open(path)
            pil_img.load()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Failed to Load Image",
                f"Could not open image file:\n{path}\n\n{exc}",
            )
            return

        self._load_pil_image(pil_img, os.path.basename(path), fit_view=True)

    def paste_image_from_clipboard(self) -> None:
        clipboard = QGuiApplication.clipboard()
        if clipboard is None:
            return
        mime = clipboard.mimeData()
        if mime is not None and mime.hasUrls():
            for url in mime.urls():
                local = url.toLocalFile()
                if local and os.path.isfile(local):
                    self.load_image_from_path(local)
                    return

        qimg = clipboard.image()
        if qimg.isNull():
            QMessageBox.information(
                self,
                "Clipboard Empty",
                "No image found in the system clipboard. Copy an image or image file first!",
            )
            return

        # Convert QImage to PIL Image via in-memory PNG buffer
        qimg_rgba = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
        width = qimg_rgba.width()
        height = qimg_rgba.height()
        ptr = qimg_rgba.bits()
        ptr.setsize(height * qimg_rgba.bytesPerLine())
        arr = np.frombuffer(ptr, np.uint8).reshape((height, qimg_rgba.bytesPerLine() // 4, 4))
        arr = arr[:, :width, :].copy()
        pil_img = Image.fromarray(arr, mode="RGBA")
        self._load_pil_image(pil_img, "Clipboard Image", fit_view=True)

    def load_sample_artwork(self) -> None:
        self.pause_simulation()
        sample = create_sample_image(320, 240)
        self._load_pil_image(sample, "Built-in Cyber-Sun Sample", fit_view=True)

    def _load_pil_image(self, pil_img: Image.Image, label: str, fit_view: bool = True) -> None:
        self.pause_simulation()
        self.current_image_name = label
        self.source_label.setText(
            f"Loaded: {label} ({pil_img.width}×{pil_img.height}px)"
        )
        self.engine.load_pil_image(
            pil_img,
            max_dim=self.res_slider.value(),
            threshold=self.thresh_slider.value(),
            invert=self.invert_chk.isChecked(),
        )
        if fit_view:
            self.canvas.fit_to_view()
        else:
            self.canvas.update()
        self._update_stats_labels()

    # --- Quantization & Settings Callbacks ---

    def _on_quantization_changed(self) -> None:
        self.pause_simulation()
        max_dim = self.res_slider.value()
        threshold = self.thresh_slider.value()
        invert = self.invert_chk.isChecked()

        self.res_val_label.setText(f"{max_dim} cells")
        self.thresh_val_label.setText(f"{threshold} / 255")

        prev_w, prev_h = self.engine.width, self.engine.height
        self.engine.apply_image_settings(
            max_dim=max_dim,
            threshold=threshold,
            invert=invert,
        )
        if (self.engine.width, self.engine.height) != (prev_w, prev_h):
            self.canvas.fit_to_view()
        else:
            self.canvas.update()
        self._update_stats_labels()

    def _on_fps_changed(self, value: int) -> None:
        self.fps_val_label.setText(f"{value} FPS")
        if self.is_playing:
            self.timer.setInterval(max(1, int(round(1000.0 / value))))

    def _on_toroidal_toggled(self, checked: bool) -> None:
        self.engine.toroidal = checked

    def _on_color_mode_changed(self, index: int) -> None:
        self.canvas.monochrome = index == 1
        self.canvas.update()

    def _on_grid_lines_toggled(self, checked: bool) -> None:
        self.canvas.show_grid_lines = checked
        self.canvas.update()

    def _on_brush_changed(self, radius: int) -> None:
        self.canvas.brush_radius = radius
        dim = 2 * radius + 1
        self.brush_val_label.setText(f"{dim} × {dim}")
        self.canvas.update()

    def fit_canvas(self) -> None:
        self.canvas.fit_to_view()

    # --- Playback Controls ---

    def toggle_play(self) -> None:
        if self.is_playing:
            self.pause_simulation()
        else:
            self.start_simulation()

    def start_simulation(self) -> None:
        self.is_playing = True
        self.play_btn.setText("⏸  Pause (P)")
        fps = self.fps_slider.value()
        self.timer.start(max(1, int(round(1000.0 / fps))))

    def pause_simulation(self) -> None:
        if not self.is_playing:
            return
        self.is_playing = False
        self.timer.stop()
        self.play_btn.setText("▶  Play (P)")

    def step_once(self) -> None:
        self.pause_simulation()
        self._advance_generation()

    def _on_timer_tick(self) -> None:
        self._advance_generation()

    def _advance_generation(self) -> None:
        self.engine.step()
        if self.is_recording_gif:
            self._capture_gif_frame()
        self.canvas.update()
        self._update_stats_labels()

    def reset_to_image(self) -> None:
        self.pause_simulation()
        self.engine.reset_to_initial()
        self.canvas.update()
        self._update_stats_labels()

    def clear_grid(self) -> None:
        self.pause_simulation()
        self.engine.clear()
        self.canvas.update()
        self._update_stats_labels()

    def _update_stats_labels(self) -> None:
        self.gen_label.setText(f"{self.engine.generation:,}")
        pop = self.engine.population
        total = max(1, self.engine.total_cells)
        pct = (pop / total) * 100.0
        self.pop_label.setText(f"{pop:,} ({pct:.1f}%)")
        self.dim_label.setText(f"{self.engine.width} × {self.engine.height}")

    # --- Export & GIF Recording ---

    def _render_export_pil_frame(self, upscale_min_px: int = 600) -> Image.Image:
        rgb = self.engine.render_rgb_frame(
            monochrome=self.canvas.monochrome,
            mono_fg=self.canvas.mono_fg,
            bg_color=self.canvas.bg_color,
        )
        pil_frame = Image.fromarray(rgb, mode="RGB")
        # Upscale with nearest-neighbor so exported PNG/GIF has crisp visible cells
        longest = max(self.engine.width, self.engine.height, 1)
        scale_factor = max(1, int(np.ceil(upscale_min_px / longest)))
        if scale_factor > 1:
            pil_frame = pil_frame.resize(
                (self.engine.width * scale_factor, self.engine.height * scale_factor),
                Image.Resampling.NEAREST,
            )
        return pil_frame

    def export_png(self) -> None:
        was_playing = self.is_playing
        self.pause_simulation()
        default_name = f"game_of_life_gen_{self.engine.generation}.png"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Snapshot as PNG",
            default_name,
            "PNG Image (*.png)",
        )
        if path:
            try:
                frame = self._render_export_pil_frame(upscale_min_px=800)
                frame.save(path, format="PNG")
            except Exception as exc:
                QMessageBox.warning(self, "Export Error", f"Failed to save PNG:\n{exc}")
        if was_playing:
            self.start_simulation()

    def toggle_record_gif(self) -> None:
        if not self.is_recording_gif:
            self.is_recording_gif = True
            self.recorded_frames = [self._render_export_pil_frame(upscale_min_px=500)]
            self.rec_gif_btn.setObjectName("recordBtnActive")
            self.rec_gif_btn.setText("■ Stop & Save (1)")
            self.rec_gif_btn.style().unpolish(self.rec_gif_btn)
            self.rec_gif_btn.style().polish(self.rec_gif_btn)
            if not self.is_playing:
                self.start_simulation()
        else:
            self.is_recording_gif = False
            self.pause_simulation()
            self.rec_gif_btn.setObjectName("")
            self.rec_gif_btn.setText("● Record GIF")
            self.rec_gif_btn.style().unpolish(self.rec_gif_btn)
            self.rec_gif_btn.style().polish(self.rec_gif_btn)

            if not self.recorded_frames:
                return

            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Recorded Animated GIF",
                "game_of_life_recording.gif",
                "GIF Animation (*.gif)",
            )
            if path:
                self._save_gif_frames(path, self.recorded_frames)
            self.recorded_frames.clear()

    def _capture_gif_frame(self) -> None:
        if len(self.recorded_frames) < 400:
            self.recorded_frames.append(self._render_export_pil_frame(upscale_min_px=500))
            self.rec_gif_btn.setText(f"■ Stop & Save ({len(self.recorded_frames)})")
        else:
            # Auto-stop at 400 frames to keep memory reasonable
            self.toggle_record_gif()

    def quick_export_gif(self) -> None:
        self.pause_simulation()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export 100-Generation Animated GIF",
            "game_of_life_100gens.gif",
            "GIF Animation (*.gif)",
        )
        if not path:
            return

        frames: list[Image.Image] = []
        for _ in range(100):
            frames.append(self._render_export_pil_frame(upscale_min_px=500))
            self.engine.step()
        self.canvas.update()
        self._update_stats_labels()
        self._save_gif_frames(path, frames)

    def _save_gif_frames(self, path: str, frames: list[Image.Image]) -> None:
        if not frames:
            return
        fps = max(1, self.fps_slider.value())
        duration_ms = max(20, int(round(1000.0 / fps)))
        try:
            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=duration_ms,
                loop=0,
            )
        except Exception as exc:
            QMessageBox.warning(self, "GIF Export Error", f"Failed to save GIF:\n{exc}")

    # Window-level drag & drop fallback
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        mime = event.mimeData()
        if mime is not None and mime.hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        mime = event.mimeData()
        if mime is not None and mime.hasUrls():
            for url in mime.urls():
                local = url.toLocalFile()
                if local:
                    self.load_image_from_path(local)
                    event.acceptProposedAction()
                    return


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

