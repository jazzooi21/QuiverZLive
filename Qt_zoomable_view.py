
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGraphicsView
from PySide6.QtGui import QPainter

class ZoomableGraphicsView(QGraphicsView):
    """
    Wheel to zoom (cursor stays fixed on the same scene point).
    Drag with left mouse to scroll (hand drag).
    Double-click to reset zoom to 100%.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep the point under cursor stable while zooming
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        # Smooth scroll/zoom behavior
        self.setRenderHint(QPainter.Antialiasing)
        # self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        # Zoom settings
        self._zoom = 0
        self._zoom_min = -20   # ~ 1/(1.25**20)
        self._zoom_max =  50   # ~ 1.25**50
        self._zoom_step = 1.25 # wheel step factor

    def wheelEvent(self, event):
        # Most mice deliver 120 units per notch; angleDelta().y() gives multiples of that
        delta = event.angleDelta().y()
        if delta == 0:
            return

        # Determine zoom direction
        zoom_in = delta > 0
        # Enforce zoom bounds (by counting steps)
        if zoom_in and self._zoom >= self._zoom_max:
            return
        if (not zoom_in) and self._zoom <= self._zoom_min:
            return

        factor = self._zoom_step if zoom_in else 1.0 / self._zoom_step
        self._zoom += 1 if zoom_in else -1
        self.scale(factor, factor)

    