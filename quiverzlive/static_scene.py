import math
import networkx as nx

from PySide6.QtWidgets import QGraphicsScene
from PySide6.QtGui    import QPen, QBrush, QColor, QPainter
from PySide6.QtCore   import QRectF, Qt, QLineF

from .constants import NODE_RADIUS, PEN_NODE, SCENE_WIDTH, SCENE_HEIGHT
from .graph_model import QuiverGraph, compute_balance
from .nx_layouts import plot_caterpillar, plot_sunshine, plot_sunshine_multicycles

class _StaticScene(QGraphicsScene):
    """Draws the graph once; mouse & key events are ignored."""

    def __init__(self, g: QuiverGraph, *args, **kw):
        super().__init__(*args, **kw)
        self._draw_graph(g)
        self.setSceneRect(self.itemsBoundingRect())

        self._current_colour_mode = "none" # default
    
     # --------------------------- public setters ------------------------------
    def set_colour_mode(self, mode: str):
        self._current_colour_mode = mode

    # suppress interaction
    # def mousePressEvent(self, _):   pass
    # def mouseMoveEvent (self, _):   pass
    # def mouseReleaseEvent(self, _): pass
    # def keyPressEvent(self,  _):    pass

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        # Solid white background
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(Qt.NoPen)
        painter.drawRect(rect)
        

    def _draw_graph(self, g: QuiverGraph):
            """
            Paint the graph exactly as recorded in `g`.

            • Uses node['pos'] when available.
            • If several parallel edges exist, draw each with a slight angle offset:
                – for up to 4 edges, use 0°, 90°, 180°, 270°,
                – for more, evenly spread 360°/n.
            • Node label replicates the logic from the main editor window.
            """
            if all(("pos" in g.nodes[n]) for n in g.nodes):
                pos = {n: tuple(g.nodes[n]["pos"]) for n in g.nodes}
            else:
                # ---------- 1. node positions (your existing logic) ----------
                cycles = nx.cycle_basis(nx.Graph(g))
                if not cycles:
                    pos = plot_caterpillar(g,
                        hsep=5 * NODE_RADIUS + 10,
                        vsep=5 * NODE_RADIUS + 10,
                    )
                elif len(cycles) == 1:
                    pos = plot_sunshine(g,
                        radius=5 * NODE_RADIUS + 10,
                        vsep=5 * NODE_RADIUS + 10,
                    )
                else:
                    pos = plot_sunshine_multicycles(g)
                    scale = 5 * NODE_RADIUS + 10
                    for k in pos:
                        pos[k] = (pos[k][0] * scale, pos[k][1] * scale)

                # store for later
                for n, (x, y) in pos.items():
                    g.nodes[n]["pos"] = (x, y)


            # ---------- 2. draw edges --------------
            drawn_pairs = {}
            for u, v, key in g.edges(keys=True):
                    
                x1, y1 = pos[u] 
                x2, y2 = pos[v]

                # 1) find how many parallel edges there are between u and v
                total = g.number_of_edges(u, v)

                # 2) sort the pair so we index correctly
                pair = tuple(sorted((u, v)))
                idx  = drawn_pairs.get(pair, 0)
                drawn_pairs[pair] = idx + 1

                # 3) compute the direction vector from u→v
                dx = x2 - x1
                dy = y2 - y1
                length = math.hypot(dx, dy)
                if length == 0:
                    # overlapping nodes: skip or draw a loop
                    continue

                # 4) perpendicular unit vector (rotated 90°)
                ux = -dy / length
                uy =  dx / length

                # 5) decide your total “spacing” between adjacent edges
                if total in [1,2]:
                    spacing = 12
                else:
                    spacing = 32 / total

                # 6) compute a centered multiplier
                offset_index = idx - (total - 1) / 2.0

                # 7) displacement along the perpendicular direction
                ox = ux * spacing * offset_index
                oy = uy * spacing * offset_index

                # 8) draw the line, shifted by (ox, oy)
                self.addLine(
                    QLineF(x1 + ox, y1 + oy, x2 + ox, y2 + oy),
                    QPen(Qt.black, 2)
                )

            # ---------- 3. draw nodes --------------
            for n, data in g.nodes(data=True):
                x, y = pos[n]
                node_type = data.get("node_type")
                
                # bugfixing purposes
                # (f"Node {n}: {node_type}", data)

                # shape
                shape_func = self.addEllipse if node_type == "gauge" else self.addRect if node_type == "flav" else None
                
                if not shape_func:
                    print("somehow idk node type")

                item = shape_func(
                    QRectF(x - NODE_RADIUS, y - NODE_RADIUS,
                        2 * NODE_RADIUS, 2 * NODE_RADIUS),
                    PEN_NODE, QBrush(Qt.white)
                )
                item.setData(0, n)

                # label text identical to editor rule
                gp_type = data.get("gp_type", "?")
                gp_rank = data.get("gp_rank", "?")
                if gp_type == "U" and node_type == "flav":
                    label_txt = f"{gp_rank}"
                else:
                    label_txt = f"{gp_type}({gp_rank})"

                label = self.addText(label_txt)
                label.setDefaultTextColor(Qt.black)
                label.setPos(
                    x - label.boundingRect().width()/2,
                    y - label.boundingRect().height()/2
                )
                tooltip = self.format_tooltip(g, n)
                label_txt = tooltip.splitlines()[0]
                item.setToolTip(tooltip)


    def format_tooltip(self, g: QuiverGraph, node_id: int) -> str:

        data = g.nodes[node_id]
        gp_type   = data.get("gp_type", "?")
        gp_rank   = data.get("gp_rank", "?")
        flav_flag = data.get("node_type", "?")

        flav_flag = "gauge" if flav_flag=="gauge" else "flav" if flav_flag in ["flav","flavour"] else "?"

        gp_name = f"{gp_type}({gp_rank})\nType: {flav_flag}"

        balance = compute_balance(g, node_id)

        output_str = f"Group: {gp_name}"
        if balance is not None:
            output_str += f"\nBalance: b = {balance}"
        return output_str

