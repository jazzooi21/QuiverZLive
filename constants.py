from PySide6.QtCore import Qt
from PySide6.QtGui import QPen, QColor

# Geometry
GRID_SCALE   = 40          # pixels between grid lines
NODE_RADIUS  = 15          # radius (or half-side) of node shapes
SCENE_WIDTH  = 800
SCENE_HEIGHT = 600

# Pens / colours
PEN_GRID_THIN  = QPen(QColor(240, 240, 240), 1)
PEN_GRID_THICK = QPen(QColor(240, 240, 240), 2)
PEN_NODE       = QPen(Qt.black, 2)