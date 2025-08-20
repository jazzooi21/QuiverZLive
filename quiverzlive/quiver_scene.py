import math
import networkx as nx
from typing import Optional

from PySide6.QtWidgets import (QGraphicsScene, QGraphicsEllipseItem,
                               QGraphicsRectItem, QGraphicsLineItem,
                               QGraphicsTextItem, QMenu, QGraphicsItem,
                               QMessageBox)
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QMouseEvent, QTransform
from PySide6.QtCore import QRectF, QLineF, Qt

from .constants import (GRID_SCALE, NODE_RADIUS, SCENE_WIDTH, SCENE_HEIGHT,
                        PEN_GRID_THIN, PEN_GRID_THICK, PEN_NODE)
from .graph_model import QuiverGraph



class QuiverScene(QGraphicsScene):
    """
    Scene that draws a snap-to-grid quiver and keeps a live NetworkX model.
    Labels are child items of their nodes so they move automatically.
    """

    # ---------------------------------- init ---------------------------------

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSceneRect(0, 0, SCENE_WIDTH, SCENE_HEIGHT)

        self.graph              = QuiverGraph()
        self.edge_items         = {}
        self.current_node_type  = "gauge"        # default ("gauge" or "flav")
        self.mode               = "add"       # default (interaction mode: "cursor" or "add")
        self._dragged_node      = None
        self._drag_offset       = None
        self._dragged_node_edges = []
        self._drag_start_node   = None
        self._drag_line         = None
        self._next_id           = 0

        

    # --------------------------- public setters ------------------------------

    def set_node_type(self, node_type: str):
        self.current_node_type = node_type
    
    def set_interaction_mode(self, mode: str):
        self.mode = mode

    # --------------------------- Qt paint events -----------------------------

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        """Highlight selected nodes with rectangles and selected edges with blue dashed lines."""
        pen = QPen(QColor(0, 0, 255), 2, Qt.DashLine)
        painter.setPen(pen)

        for item in self.selectedItems():
            if isinstance(item, (QGraphicsEllipseItem, QGraphicsRectItem)):
                painter.drawRect(item.sceneBoundingRect())
            elif isinstance(item, QGraphicsLineItem):
                painter.drawLine(item.line())

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        # Solid white background
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(Qt.NoPen)
        painter.drawRect(rect)

        # Grid
        thin, thick = [], []
        for x in range(int(rect.left() / GRID_SCALE), math.ceil(rect.right() / GRID_SCALE) + 1):
            line = QLineF(x * GRID_SCALE, rect.top(), x * GRID_SCALE, rect.bottom())
            (thick if x % 4 == 0 else thin).append(line)
        for y in range(int(rect.top() / GRID_SCALE), math.ceil(rect.bottom() / GRID_SCALE) + 1):
            line = QLineF(rect.left(), y * GRID_SCALE, rect.right(), y * GRID_SCALE)
            (thick if y % 4 == 0 else thin).append(line)

        painter.setPen(PEN_GRID_THIN);  painter.drawLines(thin)
        painter.setPen(PEN_GRID_THICK); painter.drawLines(thick)

    # ----------------------------- mouse input ------------------------------

    def mousePressEvent(self, event: QMouseEvent):
        pos = event.scenePos()
        is_on_node = any(isinstance(item, (QGraphicsEllipseItem, QGraphicsRectItem))
                        for item in self.items(pos))

        if event.button() == Qt.LeftButton:
            # Left click → Always try to drag a node if clicked on one
            for item in self.items(pos):
                if isinstance(item, (QGraphicsEllipseItem, QGraphicsRectItem)):
                    self._start_node_drag(item, pos)
                    break

        elif event.button() == Qt.RightButton:
            if self.mode == "cursor":
                # Right click → show context menu
                self._show_context_menu_at(pos, event.screenPos())

            elif self.mode == "add":
                if is_on_node:
                    # Right click on node → start edge drag
                    for item in self.items(pos):
                        if isinstance(item, (QGraphicsEllipseItem, QGraphicsRectItem)):
                            self._drag_start_node = item
                            self._drag_line = self.addLine(QLineF(pos, pos), QPen(Qt.black, 2, Qt.DashLine))
                            break
                else:
                    # Right click on empty grid → add node
                    self._add_node_at_grid(pos)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragged_node:
                self._during_node_drag(event)
        elif self._drag_line:
            self._during_edge_drag(event)
        super().mouseMoveEvent(event)



    def mouseReleaseEvent(self, event):
        if self.mode in ("cursor", "add"):
            self._on_mouse_release(event)
        super().mouseReleaseEvent(event)


    # ------------------------------- helpers --------------------------------

    # ---- drag node ----
    def _during_node_drag(self, event: QMouseEvent):
        node_id = self._dragged_node.data(0)  # stable ID

        new_center = event.scenePos() + self._drag_offset
        grid_x = round(new_center.x() / GRID_SCALE) * GRID_SCALE
        grid_y = round(new_center.y() / GRID_SCALE) * GRID_SCALE
        rect = self._dragged_node.rect()
        self._dragged_node.setPos(grid_x - rect.center().x(),
                                grid_y - rect.center().y())
        

        # Update node position in the graph model
        self.graph.nodes[node_id]['x'] = grid_x
        self.graph.nodes[node_id]['y'] = grid_y

        #  _redraw_edges_between will remove & re‐add all edges for a given pair —
        #    so just call it for every neighbor of the dragged node
        for neighbor in self.graph.neighbors(node_id):
            self._redraw_edges_between(node_id, neighbor)

    # ---- drag edge ----
    def _during_edge_drag(self, event: QMouseEvent):
        start = self._drag_line.line().p1()
        self._drag_line.setLine(QLineF(start, event.scenePos()))

    # ---- mouse release ----
    def _on_mouse_release(self, event: QMouseEvent):
        # finish node drag
        self._dragged_node = self._drag_offset = None
        self._dragged_node_edges.clear()

        # finish edge drag
        if self._drag_line:
            end_item = self._find_node_at_pos(event.scenePos())
            if end_item and end_item is not self._drag_start_node:
                self._commit_edge(self._drag_start_node, end_item)
            self.removeItem(self._drag_line)
            self._drag_line = self._drag_start_node = None

    # --------------------- node / edge creation helpers ---------------------
    
    def _add_node_at_grid(self, pos):
        gx = round(pos.x() / GRID_SCALE) * GRID_SCALE
        gy = round(pos.y() / GRID_SCALE) * GRID_SCALE

        # Check if a node already exists at this position
        for item in self.items(QRectF(gx - 1, gy - 1, 2, 2)):
            if isinstance(item, (QGraphicsEllipseItem, QGraphicsRectItem)):
                QMessageBox.warning(
                    self.views()[0],  # parent: the view showing this scene
                    "Node Exists",
                    f"A node already exists here."
                )
                return
        self._create_node(gx, gy)

    # Low-level creation (graphics + graph)
    def _create_node(self, x: int, y: int):

        # ---- graphics item ----
        if self.current_node_type == "gauge":
            shape = QGraphicsEllipseItem(QRectF(x - NODE_RADIUS, y - NODE_RADIUS,
                                                2*NODE_RADIUS, 2*NODE_RADIUS))
        else:  # flavour
            shape = QGraphicsRectItem(QRectF(x - NODE_RADIUS, y - NODE_RADIUS,
                                             2*NODE_RADIUS, 2*NODE_RADIUS))
        shape.setBrush(QBrush(Qt.white))
        shape.setPen(PEN_NODE)
        shape.setZValue(1)
        self.addItem(shape)
    
        # ---- label ----
        main_window = self._get_main_window()
        gp_type   = main_window.group_type_combo.currentText()
        gp_rank   = int(main_window.group_number_combo.currentText())
        label_txt = f"{gp_rank}" if (gp_type == "U" and self.current_node_type == "flav") else f"{gp_type}({gp_rank})"

        label = QGraphicsTextItem(label_txt, shape)  # Make label a child of the node
        label.setDefaultTextColor(Qt.black)
        label.setZValue(2)
        # Position label at the center of the node using local coordinates
        label_rect = label.boundingRect()
        label.setPos(
            shape.rect().center().x() - label_rect.width() / 2,
            shape.rect().center().y() - label_rect.height() / 2
        )
        self.addItem(label)

        # ---- graph model ----
        node_id = self._next_id
        self._next_id += 1
        shape.setData(0, node_id)          # store inside the QGraphicsItem
        self.graph.add_quiver_node(
            node_id,
            gp_type=gp_type,
            gp_rank=gp_rank,
            node_type=self.current_node_type
        )

        self.graph.nodes[node_id]['x'] = x
        self.graph.nodes[node_id]['y'] = y

        shape.setFlag(QGraphicsItem.ItemIsSelectable)

    def _commit_edge(self, shape_u, shape_v):   
        uid = shape_u.data(0)
        vid = shape_v.data(0)

        # Add to graph first
        self.graph.add_quiver_edge(uid, vid)

        # Rebuild all edges for this pair
        self._redraw_edges_between(uid, vid)    


    def _redraw_edges_between(self, u, v):
        # Ensure sorted pair (consistent indexing)
        pair = tuple(sorted((u, v)))

        # Remove old edge graphics for this pair
        if pair in self.edge_items:
            for item in self.edge_items[pair]:
                self.removeItem(item)
        self.edge_items[pair] = []

        # Get positions
        c1 = self._node_center(u)
        c2 = self._node_center(v)
        if c1 is None or c2 is None:
            return
        x1, y1 = c1.x(), c1.y()
        x2, y2 = c2.x(), c2.y()

        # Direction vector u→v
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return  # overlapping nodes

        dirx, diry = dx / length, dy / length
        ux, uy = -dy / length, dx / length

        # Total edges
        total = self.graph.number_of_edges(u, v)
        spacing = 12 if total in [0, 1, 2] else 32 / total

        # Draw each edge with updated offset
        for idx in range(total):
            offset_index = idx - (total - 1) / 2.0
            ox = ux * spacing * offset_index
            oy = uy * spacing * offset_index

            # Trim to node borders
            r = NODE_RADIUS
            x1_edge = x1 + ox + dirx * r
            y1_edge = y1 + oy + diry * r
            x2_edge = x2 + ox - dirx * r
            y2_edge = y2 + oy - diry * r

            # Draw
            edge_item = self.addLine(QLineF(x1_edge, y1_edge, x2_edge, y2_edge), QPen(Qt.black, 2))
            edge_item.setData(0, (u, v))
            edge_item.setFlag(QGraphicsItem.ItemIsSelectable)
            self.edge_items[pair].append(edge_item)


    # -------------------------- drag detection ------------------------------

    def _start_node_drag(self, item, click_pos):
        self._dragged_node = item
        node_center = item.sceneBoundingRect().center()
        self._drag_offset = node_center - click_pos
        self._dragged_node_edges = self._find_edges_for_node(item)

    # ------------------------ small helper utils ---------------------------

    def _node_id_from_pos(self, gx: int, gy: int):
        return (gx, gy)  # tuple grid coordinate is unique & hashable

    def _find_node_at_pos(self, pos) -> Optional[QGraphicsEllipseItem]:
        for it in self.items(pos):
            if isinstance(it, (QGraphicsEllipseItem, QGraphicsRectItem)):
                return it
        return None

    def _find_edges_for_node(self, node_item):
        node_center = node_item.sceneBoundingRect().center()
        edges = []
        for it in self.items():
            if isinstance(it, QGraphicsLineItem):
                line = it.line()
                p1, p2 = line.p1(), line.p2()
                if QLineF(p1, node_center).length() < NODE_RADIUS+2:
                    edges.append((it, "p1"))
                elif QLineF(p2, node_center).length() < NODE_RADIUS+2:
                    edges.append((it, "p2"))
        return edges

    def _get_main_window(self):
        return self.views()[0].window() if self.views() else None
    
    def _node_center(self, node_id):
        # find the graphics item whose data(0) == node_id
        for item in self.items():
            if isinstance(item, (QGraphicsEllipseItem, QGraphicsRectItem)) \
            and item.data(0) == node_id:
                return item.sceneBoundingRect().center()
        return None
    
    
    # -------------------------- clear all function -----------------------------
    def clear(self):
        super().clear()        
        self.graph.clear()

    # -------------------------- context menu for nodes/edges ------------------

    def _show_context_menu_at(self, scene_pos, screen_pos):
        if self.mode != "cursor":
            return
    
        item = self.itemAt(scene_pos, QTransform())
        while item and isinstance(item, QGraphicsTextItem):
            item = item.parentItem()
        menu = QMenu()

        if isinstance(item, (QGraphicsEllipseItem, QGraphicsRectItem)):
            delete = menu.addAction("Delete Node")
            action = menu.exec(screen_pos)
            if action is delete:
                self._remove_node(item)

        elif isinstance(item, QGraphicsLineItem):
            delete = menu.addAction("Delete Edge(s)")
            action = menu.exec(screen_pos)
            if action is delete:
                self._remove_edge(item)


    # -------------------------- node/edge removal ----------------------------
    def _remove_node(self, node_item):
        # 1) graphically remove all its incident edges
        nid = node_item.data(0)
        for it in list(self.items()):
            if isinstance(it, QGraphicsLineItem):
                u, v = it.data(0)
                if u == nid or v == nid:
                    self.removeItem(it)
        # remove the node shape (its child label is auto-removed)
        self.removeItem(node_item)
        # remove from the MultiGraph (also drops all edges!)
        self.graph.remove_node(nid)

    def _remove_edge(self, edge_item):
        u, v = edge_item.data(0)
        if self.graph.has_edge(u, v):
            self.graph.remove_edge(u, v)
            self._redraw_edges_between(u, v)


    # -------------------------- window close event ---------------------------
    def closeEvent(self, event):
        """
        When the window closes, purge the scene so every QGraphicsItem
        is released, then let Qt finish normal teardown.
        """
        self.scene.clear()          # remove all nodes/edges/labels
        self.view.setScene(None)    # break view ↔ scene link
        super().closeEvent(event)   # continue default handling
