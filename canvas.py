from __future__ import annotations

from typing import Callable
import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QWheelEvent,
)
from PyQt6.QtWidgets import QWidget

from engine import LifeEngine


class GridCanvas(QWidget):
    """Interactive Game of Life viewport with zoom, pan, draw/erase, and drag-drop."""

    cellsEdited = pyqtSignal()
    imageDropped = pyqtSignal(str)

    def __init__(self, engine: LifeEngine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Display settings
        self.monochrome: bool = False
        self.show_grid_lines: bool = True
        self.brush_radius: int = 0
        self.bg_color: tuple[int, int, int] = (18, 18, 24)
        self.mono_fg: tuple[int, int, int] = (0, 235, 145)

        # Viewport transform: screen_pos = grid_pos * scale + offset
        self.scale: float = 4.0
        self.offset: QPointF = QPointF(20.0, 20.0)
        self._needs_initial_fit: bool = True

        # Interaction state
        self._space_pressed: bool = False
        self._panning: bool = False
        self._pan_start_mouse: QPointF = QPointF()
        self._pan_start_offset: QPointF = QPointF()

        self._drawing: bool = False
        self._draw_alive_mode: bool = True
        self._last_cell: tuple[int, int] | None = None
        self._hover_cell: tuple[int, int] | None = None

    def fit_to_view(self) -> None:
        """Fit and center the grid comfortably inside the widget viewport."""
        w = max(1, self.width())
        h = max(1, self.height())
        gw = max(1, self.engine.width)
        gh = max(1, self.engine.height)

        padding = 32.0
        avail_w = max(40.0, w - 2 * padding)
        avail_h = max(40.0, h - 2 * padding)

        self.scale = min(avail_w / gw, avail_h / gh)
        self.scale = max(0.5, min(self.scale, 64.0))

        total_w = gw * self.scale
        total_h = gh * self.scale
        self.offset = QPointF((w - total_w) * 0.5, (h - total_h) * 0.5)
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        if self._needs_initial_fit and self.width() > 50 and self.height() > 50:
            self._needs_initial_fit = False
            self.fit_to_view()

    def screen_to_grid(self, pos: QPointF) -> tuple[int, int]:
        gx = int(np.floor((pos.x() - self.offset.x()) / self.scale))
        gy = int(np.floor((pos.y() - self.offset.y()) / self.scale))
        return gx, gy

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(11, 11, 15))

        gw, gh = self.engine.width, self.engine.height
        if gw <= 0 or gh <= 0:
            return

        # Render engine state to contiguous RGB888 buffer
        rgb = np.ascontiguousarray(
            self.engine.render_rgb_frame(
                monochrome=self.monochrome,
                mono_fg=self.mono_fg,
                bg_color=self.bg_color,
            )
        )
        bytes_per_line = gw * 3
        qimg = QImage(
            rgb.data,
            gw,
            gh,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )

        target_rect = QRectF(
            self.offset.x(),
            self.offset.y(),
            gw * self.scale,
            gh * self.scale,
        )

        # Draw crisp nearest-neighbor cells
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(target_rect, qimg)

        # Optional grid lines when cells are large enough on screen
        if self.show_grid_lines and self.scale >= 4.5:
            grid_alpha = min(90, int((self.scale - 4.0) * 14))
            pen = QPen(QColor(255, 255, 255, grid_alpha))
            pen.setWidth(1)
            painter.setPen(pen)

            # Only draw lines within the visible viewport
            min_gx = max(0, int(np.floor(-self.offset.x() / self.scale)))
            max_gx = min(gw, int(np.ceil((self.width() - self.offset.x()) / self.scale)))
            min_gy = max(0, int(np.floor(-self.offset.y() / self.scale)))
            max_gy = min(gh, int(np.ceil((self.height() - self.offset.y()) / self.scale)))

            top_y = max(target_rect.top(), 0.0)
            bot_y = min(target_rect.bottom(), float(self.height()))
            left_x = max(target_rect.left(), 0.0)
            right_x = min(target_rect.right(), float(self.width()))

            for gx in range(min_gx, max_gx + 1):
                x = self.offset.x() + gx * self.scale
                painter.drawLine(QPointF(x, top_y), QPointF(x, bot_y))

            for gy in range(min_gy, max_gy + 1):
                y = self.offset.y() + gy * self.scale
                painter.drawLine(QPointF(left_x, y), QPointF(right_x, y))

        # Subtle border around the grid
        border_pen = QPen(QColor(80, 90, 115, 180))
        border_pen.setWidth(1)
        painter.setPen(border_pen)
        painter.drawRect(target_rect)

        # Brush cursor preview
        if self._hover_cell is not None and not self._panning:
            hx, hy = self._hover_cell
            if 0 <= hx < gw and 0 <= hy < gh:
                r = self.brush_radius
                x0 = max(0, hx - r)
                y0 = max(0, hy - r)
                x1 = min(gw, hx + r + 1)
                y1 = min(gh, hy + r + 1)
                cursor_rect = QRectF(
                    self.offset.x() + x0 * self.scale,
                    self.offset.y() + y0 * self.scale,
                    (x1 - x0) * self.scale,
                    (y1 - y0) * self.scale,
                )
                cursor_pen = QPen(QColor(255, 255, 255, 180))
                cursor_pen.setWidth(1)
                painter.setPen(cursor_pen)
                painter.drawRect(cursor_rect)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return

        factor = 1.18 if delta > 0 else (1.0 / 1.18)
        new_scale = max(0.5, min(80.0, self.scale * factor))
        if abs(new_scale - self.scale) < 1e-6:
            return

        mouse_pos = event.position()
        # Keep the grid coordinate under mouse_pos stationary
        grid_x = (mouse_pos.x() - self.offset.x()) / self.scale
        grid_y = (mouse_pos.y() - self.offset.y()) / self.scale

        self.scale = new_scale
        self.offset = QPointF(
            mouse_pos.x() - grid_x * self.scale,
            mouse_pos.y() - grid_y * self.scale,
        )
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.setFocus()
        pos = event.position()

        if (
            event.button() == Qt.MouseButton.MiddleButton
            or (event.button() == Qt.MouseButton.LeftButton and self._space_pressed)
        ):
            self._panning = True
            self._pan_start_mouse = pos
            self._pan_start_offset = QPointF(self.offset)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._drawing = True
            self._draw_alive_mode = event.button() == Qt.MouseButton.LeftButton
            gx, gy = self.screen_to_grid(pos)
            self._last_cell = (gx, gy)
            if self.engine.paint_cells(
                gx,
                gy,
                draw_alive=self._draw_alive_mode,
                brush_radius=self.brush_radius,
                fallback_color=self.mono_fg,
            ):
                self.cellsEdited.emit()
                self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        gx, gy = self.screen_to_grid(pos)
        self._hover_cell = (gx, gy)

        if self._panning:
            delta = pos - self._pan_start_mouse
            self.offset = self._pan_start_offset + delta
            self.update()
            return

        if self._drawing:
            last_x, last_y = self._last_cell if self._last_cell is not None else (gx, gy)
            changed = False
            for ix, iy in self._iter_line(last_x, last_y, gx, gy):
                if self.engine.paint_cells(
                    ix,
                    iy,
                    draw_alive=self._draw_alive_mode,
                    brush_radius=self.brush_radius,
                    fallback_color=self.mono_fg,
                ):
                    changed = True
            self._last_cell = (gx, gy)
            if changed:
                self.cellsEdited.emit()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._panning and event.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.LeftButton,
        ):
            self._panning = False
            self.setCursor(
                Qt.CursorShape.OpenHandCursor
                if self._space_pressed
                else Qt.CursorShape.ArrowCursor
            )
        if self._drawing and event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        ):
            self._drawing = False
            self._last_cell = None

    def leaveEvent(self, event) -> None:  # noqa: ANN001
        self._hover_cell = None
        self.update()

    def set_space_pan(self, pressed: bool) -> None:
        self._space_pressed = pressed
        if not self._panning:
            self.setCursor(
                Qt.CursorShape.OpenHandCursor
                if pressed
                else Qt.CursorShape.ArrowCursor
            )

    @staticmethod
    def _iter_line(x0: int, y0: int, x1: int, y1: int):
        """Bresenham line generator so fast mouse strokes don't skip cells."""
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            yield x0, y0
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    # Drag-and-drop support
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        mime = event.mimeData()
        if mime is not None and (mime.hasUrls() or mime.hasImage()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        mime = event.mimeData()
        if mime is None:
            return
        if mime.hasUrls():
            for url in mime.urls():
                local_path = url.toLocalFile()
                if local_path:
                    self.imageDropped.emit(local_path)
                    event.acceptProposedAction()
                    return

