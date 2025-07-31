import sys
import math
import networkx as nx
from PySide6.QtWidgets import (
    QApplication, QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QGraphicsRectItem,
    QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QMainWindow,
    QComboBox, QGraphicsTextItem, QGraphicsLineItem
)
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QMouseEvent
from PySide6.QtCore import QRectF, Qt, QLineF, QPointF

GRID_SCALE = 40
NODE_RADIUS = 15


class QuiverScene(QGraphicsScene):
    """Graphics scene that also keeps a parallel NetworkX MultiGraph."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.draw_background_lines = True
        self.setSceneRect(0, 0, 800, 600)
        self.current_node_type = "gauge"  # default

        # --- NetworkX data structures ---
        self.graph = nx.MultiGraph()
        self._node_counter = 0  # unique integer IDs for each new node
        self._item_to_node_id: dict[QGraphicsEllipseItem | QGraphicsRectItem, int] = {}

    # ---------------------------------------------------------------------
    # Public helpers -------------------------------------------------------
    # ---------------------------------------------------------------------
    def set_node_type(self, node_type: str):
        self.current_node_type = node_type

    # ---------------------------------------------------------------------
    # Qt paint routines ----------------------------------------------------
    # ---------------------------------------------------------------------
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        # Fill background
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.drawRect(rect)

        if not self.draw_background_lines:
            return

        lines, thick_lines = [], []

        for x in range(int(rect.left() / GRID_SCALE), math.ceil(rect.right() / GRID_SCALE) + 1):
            line = QLineF(x * GRID_SCALE, rect.top(), x * GRID_SCALE, rect.bottom())
            (thick_lines if x % 4 == 0 else lines).append(line)

        for y in range(int(rect.top() / GRID_SCALE), math.ceil(rect.bottom() / GRID_SCALE) + 1):
            line = QLineF(rect.left(), y * GRID_SCALE, rect.right(), y * GRID_SCALE)
            (thick_lines if y % 4 == 0 else lines).append(line)

        painter.setPen(QPen(QColor(240, 240, 240), 1))
        painter.drawLines(lines)

        painter.setPen(QPen(QColor(240, 240, 240), 2))
        painter.drawLines(thick_lines)

    # ---------------------------------------------------------------------
    # Event handling -------------------------------------------------------
    # ---------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._handle_left_press(event)
        elif event.button() == Qt.RightButton:
            self._handle_right_press(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if hasattr(self, "_dragged_node") and self._dragged_node:
            self._handle_node_drag(event)
        elif hasattr(self, "_drag_line") and self._drag_line:
            self._handle_edge_drag(event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._handle_release(event)
        super().mouseReleaseEvent(event)

    # ---------------------------------------------------------------------
    # Internal helpers -----------------------------------------------------
    # ---------------------------------------------------------------------
    def _handle_left_press(self, event):
        pos = event.scenePos()
        for item in self.items(pos):
            if isinstance(item, (QGraphicsEllipseItem, QGraphicsRectItem)):
                self._start_drag_node(item, pos)
                break
        else:
            self._add_node_at_grid(event.scenePos())
            self._clear_drag_state()

    def _handle_right_press(self, event):
        self._drag_start_node = None
        pos = event.scenePos()
        for item in self.items(pos):
            if isinstance(item, (QGraphicsEllipseItem, QGraphicsRectItem)):
                self._drag_start_node = item
                self._drag_line = self.addLine(QLineF(pos, pos), QPen(Qt.black, 2, Qt.DashLine))
                break

    def _handle_node_drag(self, event):
        new_center = event.scenePos() + getattr(self, "_drag_offset", QPointF(0, 0))
        grid_x = round(new_center.x() / GRID_SCALE) * GRID_SCALE
        grid_y = round(new_center.y() / GRID_SCALE) * GRID_SCALE
        node_rect = self._dragged_node.rect() if hasattr(self._dragged_node, "rect") else QRectF(0, 0, 0, 0)
        node_center_offset = node_rect.center()
        self._dragged_node.setPos(grid_x - node_center_offset.x(),
                                  grid_y - node_center_offset.y())
        # Move the text next to the node as it moves
        if hasattr(self, "_dragged_node_text") and self._dragged_node_text:
            self._dragged_node_text.setPos(grid_x + NODE_RADIUS + 5, grid_y - NODE_RADIUS * 1.5 / 2)
        if hasattr(self, "_dragged_node_edges"):
            node_shape = self._dragged_node
            new_node_center = node_shape.sceneBoundingRect().center()
            for edge, endpoint in self._dragged_node_edges:
                line = edge.line()
                other_point = line.p2() if endpoint == "p1" else line.p1()
                if endpoint == "p1":
                    edge.setLine(QLineF(new_node_center, other_point))
                else:
                    edge.setLine(QLineF(other_point, new_node_center))

    def _handle_edge_drag(self, event):
        start = self._drag_line.line().p1()
        self._drag_line.setLine(QLineF(start, event.scenePos()))

    def _handle_release(self, event):
        if hasattr(self, "_dragged_node") and self._dragged_node:
            self._dragged_node = None
            self._drag_offset = None
        if hasattr(self, "_dragged_node_text") and self._dragged_node_text:
            self._dragged_node_text = None
        if hasattr(self, "_dragged_node_edges") and self._dragged_node_edges:
            self._dragged_node_edges = []
        if hasattr(self, "_drag_line") and self._drag_line:
            start_node = getattr(self, "_drag_start_node", None)
            end_node = None
            pos = event.scenePos()
            for item in self.items(pos):
                if isinstance(item, (QGraphicsEllipseItem, QGraphicsRectItem)) and item is not start_node:
                    end_node = item
                    break
            self.removeItem(self._drag_line)
            self._drag_line = None
            self._drag_start_node = None
            if start_node and end_node:
                start_center = start_node.rect().center() + start_node.pos() if hasattr(start_node, "rect") else start_node.pos()
                end_center = end_node.rect().center() + end_node.pos() if hasattr(end_node, "rect") else end_node.pos()
                self.addLine(QLineF(start_center, end_center), QPen(Qt.black, 2))

                # --- Add edge to NetworkX graph ---
                start_id = self._item_to_node_id.get(start_node)
                end_id = self._item_to_node_id.get(end_node)
                if start_id is not None and end_id is not None:
                    self.graph.add_edge(start_id, end_id)

    def _start_drag_node(self, item, pos):
        self._dragged_node = item
        node_center = item.sceneBoundingRect().center()
        self._drag_offset = node_center - pos
        self._dragged_node_text = None
        # Find the text item associated with this node (by proximity)
        for text_item in self.items():
            if isinstance(text_item, QGraphicsTextItem):
                text_rect = text_item.sceneBoundingRect()
                expected_x = node_center.x() + NODE_RADIUS + 5
                expected_y = node_center.y() - NODE_RADIUS * 1.5 / 2
                if abs(text_rect.left() - expected_x) < 2 and abs(text_rect.top() - expected_y) < 2:
                    self._dragged_node_text = text_item
                    break
        self._dragged_node_edges = []
        for edge_item in self.items():
            if isinstance(edge_item, QGraphicsLineItem):
                line = edge_item.line()
                p1 = edge_item.mapToScene(line.p1())
                p2 = edge_item.mapToScene(line.p2())
                if QLineF(p1, node_center).length() < NODE_RADIUS + 2:
                    self._dragged_node_edges.append((edge_item, "p1"))
                elif QLineF(p2, node_center).length() < NODE_RADIUS + 2:
                    self._dragged_node_edges.append((edge_item, "p2"))

    def _clear_drag_state(self):
        self._dragged_node = None
        self._dragged_node_text = None
        self._dragged_node_edges = []

    # ------------------------------------------------------------------
    # Node/edge creation helpers ---------------------------------------
    # ------------------------------------------------------------------
    def _add_node_at_grid(self, pos):
        grid_x = round(pos.x() / GRID_SCALE) * GRID_SCALE
        grid_y = round(pos.y() / GRID_SCALE) * GRID_SCALE
        self.add_node(grid_x, grid_y)

    def add_node(self, x: float, y: float):
        """Create a graphics item *and* a NetworkX node with attributes."""
        color = Qt.white
        if self.current_node_type == "gauge":
            node_item = QGraphicsEllipseItem(
                QRectF(x - NODE_RADIUS, y - NODE_RADIUS, NODE_RADIUS * 2, NODE_RADIUS * 2)
            )
        else:
            node_item = QGraphicsRectItem(
                QRectF(x - NODE_RADIUS, y - NODE_RADIUS, NODE_RADIUS * 2, NODE_RADIUS * 2)
            )
        node_item.setBrush(QBrush(color))
        node_item.setPen(QPen(Qt.black, 2))
        node_item.setZValue(1)
        self.addItem(node_item)

        # Display Lie group text next to the node
        main_window = self.views()[0].window() if self.views() else None
        if main_window:
            group_type = main_window.group_type_combo.currentText()
            group_number = main_window.group_number_combo.currentText()
            if group_type == "U" and self.current_node_type == "flavour":
                group_text = f"{group_number}"
            else:
                group_text = f"{group_type}({group_number})"
        else:
            # Fallback defaults if somehow no window found
            group_type = "U"
            group_number = "1"
            group_text = f"{group_type}({group_number})"

        text_item = QGraphicsTextItem(group_text)
        text_item.setDefaultTextColor(Qt.black)
        text_item.setZValue(2)
        text_item.setPos(x + NODE_RADIUS + 5, y - NODE_RADIUS * 1.5 / 2)
        self.addItem(text_item)

        # --- Add node to NetworkX graph ---
        node_id = self._node_counter
        self._node_counter += 1
        self._item_to_node_id[node_item] = node_id
        self.graph.add_node(
            node_id,
            gp_type=group_type,
            gp_rank=int(group_number),
            flav_gauge=self.current_node_type,
            x=x,
            y=y,
        )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuiverZLive")

        # Scene and view
        self.scene = QuiverScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)

        # Side panel
        self.panel = QWidget()
        self.panel_layout = QVBoxLayout(self.panel)
        self.panel_layout.setAlignment(Qt.AlignTop)

        self.label = QLabel("Node Type:")
        self.gauge_button = QPushButton("Gauge")
        self.flavour_button = QPushButton("Flavour")

        self.gauge_button.clicked.connect(self.set_gauge_node)
        self.flavour_button.clicked.connect(self.set_flavour_node)

        self.panel_layout.addWidget(self.label)
        self.panel_layout.addWidget(self.gauge_button)
        self.panel_layout.addWidget(self.flavour_button)

        # Add a top bar (menu bar)
        self.menu_bar = self.menuBar()
        file_menu = self.menu_bar.addMenu("File")
        edit_menu = self.menu_bar.addMenu("Edit")
        view_menu = self.menu_bar.addMenu("View")
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        clear_action = edit_menu.addAction("Clear All")
        clear_action.triggered.connect(self.scene.clear)

        # Lie group selection widgets
        self.group_type_label = QLabel("Group Type:")
        self.group_type_combo = QComboBox()
        self.group_type_combo.addItems(["U", "SU", "SO", "USp"])

        self.group_number_label = QLabel("Group Number:")
        self.group_number_combo = QComboBox()
        self.group_number_combo.addItems([str(i) for i in range(1, 13)])

        self.group_type_combo.currentTextChanged.connect(self.update_group_number_even)

        self.flavour_button.clicked.connect(self.update_group_type_for_flavour)
        self.gauge_button.clicked.connect(self.update_group_type_for_gauge)

        self.panel_layout.addWidget(self.group_type_label)
        self.panel_layout.addWidget(self.group_type_combo)
        self.panel_layout.addWidget(self.group_number_label)
        self.panel_layout.addWidget(self.group_number_combo)

        # Set initial state
        self.set_gauge_node()

        # Main layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.addWidget(self.view)
        main_layout.addWidget(self.panel)

        self.setCentralWidget(main_widget)
        self.resize(1000, 650)

    # ------------------------------------------------------------------
    # Button callbacks -------------------------------------------------
    # ------------------------------------------------------------------
    def set_gauge_node(self):
        self.scene.set_node_type("gauge")
        self.gauge_button.setEnabled(False)
        self.flavour_button.setEnabled(True)

    def set_flavour_node(self):
        self.scene.set_node_type("flavour")
        self.gauge_button.setEnabled(True)
        self.flavour_button.setEnabled(False)

    # ------------------------------------------------------------------
    # Lie group UX helpers --------------------------------------------
    # ------------------------------------------------------------------
    def update_group_number_even(self):
        group_type = self.group_type_combo.currentText()
        for i in range(self.group_number_combo.count()):
            number = int(self.group_number_combo.itemText(i))
            is_odd = number % 2 == 1
            if group_type == "USp" and is_odd:
                self.group_number_combo.model().item(i).setEnabled(False)
            else:
                self.group_number_combo.model().item(i).setEnabled(True)

    def update_group_type_for_flavour(self):
        # Grey out (disable) SU when flavour node is selected
        for i in range(self.group_type_combo.count()):
            item_text = self.group_type_combo.itemText(i)
            self.group_type_combo.model().item(i).setEnabled(item_text != "SU")
        # If SU is currently selected, switch to another enabled group
        if self.group_type_combo.currentText() == "SU":
            for i in range(self.group_type_combo.count()):
                if self.group_type_combo.model().item(i).isEnabled():
                    self.group_type_combo.setCurrentIndex(i)
                    break

    def update_group_type_for_gauge(self):
        # Enable all group types when gauge node is selected
        for i in range(self.group_type_combo.count()):
            self.group_type_combo.model().item(i).setEnabled(True)


# ----------------------------------------------------------------------
# Entry point -----------------------------------------------------------
# ----------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
