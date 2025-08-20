from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from PySide6.QtCore import Qt, QPointF, QRectF, QMarginsF, QTimer
from PySide6.QtGui import QPainter, QPen, QBrush, QImage, QPixmap, QPainterPath, QFont, QFontMetrics, QColor, QTransform
from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsPathItem, QGraphicsItem, QMenu, QMessageBox,
                                QDialog, QVBoxLayout, QScrollArea, QWidget, QHBoxLayout,
                                    QPushButton, QLabel)

from .static_scene import _StaticScene
from .graph_model import QuiverGraph, is_isomorphic

def compute_base_pos(full_qg: QuiverGraph) -> Dict[int, Tuple[float, float]]:
    """
    Compute and store positions for the FULL graph once (offscreen),
    then read them back from node['pos'].
    Returns {original_node_index -> (x, y)} using the graph’s node ids.
    """
    # Trigger layout & side-effect population of g.nodes[n]['pos']
    _ = _StaticScene(full_qg)

    base: Dict[int, Tuple[float, float]] = {}
    for n, d in full_qg.nodes(data=True):
        x, y = d["pos"]
        base[int(n)] = (float(x), float(y))
    return base


def make_subquiver_with_fixed_pos_from_qg(
    full_qg: QuiverGraph,
    v: np.ndarray,
    base_pos: Dict[int, Tuple[float, float]],
) -> Optional[QuiverGraph]:
    """
    Keep original node IDs by inducing a subgraph on the nodes with v[i] > 0.
    If nothing remains, return None.
    """
    kept = [int(i) for i, r in enumerate(v) if r > 0 and int(i) in full_qg.nodes]
    if not kept:
        return None

    # Induced subgraph with original IDs & edge attrs intact
    qg_sub = full_qg.subgraph(kept).copy()

    # Restore fixed positions (same IDs, so this is 1:1)
    for n in qg_sub.nodes:
        if n in base_pos:
            qg_sub.nodes[n]["pos"] = base_pos[n]

    return qg_sub



# ─────────────────────────── rendering helpers ───────────────────────────

def quiver_to_pixmap(
    qg: QuiverGraph,
    *,
    pad: int = 4,                       # small breathing room
    supersample: int = 2,               # crisp text/lines
    bg: Qt.GlobalColor | int = Qt.white,
    source_rect: Optional[QRectF] = None,
) -> QPixmap:
    """Render a QuiverGraph tightly cropped to its content (no fixed w×h)."""
    scene = _StaticScene(qg)

    rect = source_rect
    if rect is None:
        rect = scene.itemsBoundingRect().adjusted(-pad, -pad, pad, pad)

    W = max(1, int(round(rect.width())))
    H = max(1, int(round(rect.height())))

    img = QImage(W * supersample, H * supersample, QImage.Format_ARGB32)
    img.fill(bg)

    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)
    scene.render(p, QRectF(0, 0, img.width(), img.height()), rect)
    p.end()

    pm = QPixmap.fromImage(img)
    pm.setDevicePixelRatio(supersample)  # displays as W×H but high-res
    return pm



def render_leaf_thumbnail_and_subquivers(
    full_qg: QuiverGraph,
    leaf: List[Tuple[int, ...]],
    base_pos: Dict[int, Tuple[float, float]],
    *,
    idx2node: list,
    pad: int = 16,
    box_pad: int = 10,
    box_pen_w: int = 2,
    supersample: int = 2,
    bg: Qt.GlobalColor | int = Qt.white,
) -> Tuple[QPixmap, List[QuiverGraph]]:
    """Build the leaf thumbnail AND return the list of sub-quivers used (one per component)."""
    sub_qgs: List[QuiverGraph] = []
    comps: List[QPixmap] = []

    for vec in leaf:
        v = np.array(vec, dtype=int)
        kept_nodes = [idx2node[i] for i, r in enumerate(v) if r > 0]
        if not kept_nodes:
            continue
        qg_sub = full_qg.subgraph(kept_nodes).copy()
        for n in qg_sub.nodes:
            if n in base_pos:
                qg_sub.nodes[n]["pos"] = base_pos[n]
        _apply_leaf_ranks_inplace(qg_sub, v, idx2node)

        pm = quiver_to_pixmap(qg_sub, pad=6, supersample=supersample, bg=bg)
        sub_qgs.append(qg_sub)
        comps.append(pm)

    # Reuse your leaf composition (⊗, black box etc.)
    if not comps:
        # fall back to your “empty quiver” pixmap
        empty_pm = leaf_to_pixmap_keep_pos(
            full_qg, leaf, base_pos,
            idx2node=idx2node,
            pad=pad, box_pad=box_pad, box_pen_w=box_pen_w,
            supersample=supersample, bg=bg,
        )
        return empty_pm, []

    # Compose using your current function so visuals stay identical:
    pm = leaf_to_pixmap_keep_pos(
        full_qg, leaf, base_pos,
        idx2node=idx2node,
        pad=pad, box_pad=box_pad, box_pen_w=box_pen_w,
        supersample=supersample, bg=bg,
    )
    return pm, sub_qgs


def leaf_to_pixmap_keep_pos(
    full_qg: QuiverGraph,
    leaf: List[Tuple[int, ...]],
    base_pos: Dict[int, Tuple[float, float]],
    *,
    idx2node: list,
    pad: int = 16,
    box_pad: int = 10,
    box_pen_w: int = 2,
    supersample: int = 2,
    bg: Qt.GlobalColor | int = Qt.white,
) -> QPixmap:
    # -- build tight-cropped component pixmaps (no fixed per_w/per_h) --
    comps: List[QPixmap] = []
    for vec in leaf:
        v = np.array(vec, dtype=int)
        kept_nodes = [idx2node[i] for i, r in enumerate(v) if r > 0]
        if not kept_nodes:
            continue

        qg_sub = full_qg.subgraph(kept_nodes).copy()

        # keep original positions (fixed across leaves)
        for n in qg_sub.nodes:
            if n in base_pos:
                qg_sub.nodes[n]["pos"] = base_pos[n]

        _apply_leaf_ranks_inplace(qg_sub, v, idx2node)

        # tight, cropped pixmap for this subquiver
        pm = quiver_to_pixmap(qg_sub, pad=6, supersample=2, bg=bg)
        comps.append(pm)

    # -- empty leaf (no vectors or everything cleaned away) --
    if not leaf or not comps:
        W, H = 300, 180
        img = QImage(W * supersample, H * supersample, QImage.Format_ARGB32)
        img.fill(bg)
        p = QPainter(img); p.setRenderHint(QPainter.Antialiasing)
        p.scale(supersample, supersample)
        p.setPen(QPen(Qt.black, box_pen_w)); p.setBrush(Qt.NoBrush)
        p.drawRect(1, 1, W - 2, H - 2)
        f = QFont(); 
        if f.pixelSize() > 0: f.setPixelSize(int(f.pixelSize() * 4))
        else:                  f.setPointSizeF(f.pointSizeF() * 4)
        p.setFont(f); p.setPen(Qt.black)
        p.drawText(QRectF(0, 0, W, H), Qt.AlignCenter, "empty\nquiver")
        p.end()
        pm = QPixmap.fromImage(img); pm.setDevicePixelRatio(supersample)
        return pm

    # -- ⊗ font metrics --
    font = QFont()
    if font.pixelSize() > 0: font.setPixelSize(int(font.pixelSize() * 7))
    else:                    font.setPointSizeF(font.pointSizeF() * 7)
    font.setBold(True)
    fm = QFontMetrics(font)
    sym_w = fm.horizontalAdvance("⊗")
    sym_h = fm.height()
    sep_w = sym_w + 12  # padding around the symbol

    # -- content dimensions = sum tight component widths + separators --
    total_inner_w = sum(pm.width() for pm in comps) + (len(comps) - 1) * sep_w
    total_inner_h = max(max(pm.height() for pm in comps), sym_h)  # include ⊗ height

    # -- outer black box size fits content exactly (+ pad) --
    W = total_inner_w + 2 * (pad + box_pad)
    H = total_inner_h + 2 * (pad + box_pad)

    img = QImage(W * supersample, H * supersample, QImage.Format_ARGB32)
    img.fill(bg)

    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)
    p.scale(supersample, supersample)

    # black frame
    p.setPen(QPen(Qt.black, box_pen_w)); p.setBrush(Qt.NoBrush)
    p.drawRect(1, 1, W - 2, H - 2)

    # pack components; each is tight, so no whitespace
    x = pad + box_pad
    y_top = pad + box_pad
    content_h = total_inner_h

    for i, pm in enumerate(comps):
        # Optional safety: normalize DPR to 1 before composition
        if pm.devicePixelRatio() != 1.0:
            pm2 = QPixmap(pm)             # copy
            pm2.setDevicePixelRatio(1.0)  # draw at logical size
            comp = pm2
        else:
            comp = pm

        y = y_top + (content_h - comp.height()) / 2.0
        p.drawPixmap(int(x), int(y), comp)
        x += comp.width()

        if i != len(comps) - 1:
            p.setPen(Qt.black); p.setFont(font)
            p.drawText(QRectF(x, y_top, sep_w, content_h), Qt.AlignCenter, "⊗")
            x += sep_w

    p.end()
    out = QPixmap.fromImage(img)
    out.setDevicePixelRatio(supersample)  # crisp on-screen
    return out


# ---------- geometry helpers for accurate arrow endpoints on black box ----------
pen = QPen(Qt.black, 2.0)
pen.setCapStyle(Qt.FlatCap)
pen.setJoinStyle(Qt.MiterJoin)

stroke_w = 2.0   # MUST match box_pen_w used inside the node thumbnail

def _edge_points(r: QRectF):
    return [
        (QPointF(r.left(),  r.top()),    QPointF(r.right(), r.top()   )),  # top
        (QPointF(r.right(), r.top()),    QPointF(r.right(), r.bottom())),  # right
        (QPointF(r.right(), r.bottom()), QPointF(r.left(),  r.bottom())),  # bottom
        (QPointF(r.left(),  r.bottom()), QPointF(r.left(),  r.top()   )),  # left
    ]

def _seg_intersection(p1, p2, q1, q2):
    x1,y1 = p1.x(), p1.y(); x2,y2 = p2.x(), p2.y()
    x3,y3 = q1.x(), q1.y(); x4,y4 = q2.x(), q2.y()
    den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(den) < 1e-12:
        return (False, QPointF(), 0.0)
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / den
    u = ((x1-x3)*(y1-y2) - (y1-y3)*(x1-x2)) / den
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (True, QPointF(x1 + t*(x2-x1), y1 + t*(y2-y1)), t)
    return (False, QPointF(), 0.0)

def _anchor_on_rect(p_from: QPointF, p_to: QPointF, rect: QRectF) -> QPointF:
    hits = []
    for a, b in _edge_points(rect):
        ok, pt, t = _seg_intersection(p_from, p_to, a, b)
        if ok:
            hits.append((t, pt))
    if not hits:
        return rect.center()
    # choose intersection closest to p_to
    hits.sort(key=lambda x: abs(x[0] - 1.0))
    return hits[0][1]


# ---------- Node & Edge items ----------

class NodeItem(QGraphicsPixmapItem):
    """Pixmap node that can be dragged; holds sub-quivers; hover highlight; dblclick to open."""
    def __init__(self, pixmap: QPixmap, sub_quivers: List[QuiverGraph], open_callback=None):
        super().__init__(pixmap)
        self.edges: list[EdgeItem] = []
        self.sub_quivers = sub_quivers          # 0..m-1
        self.open_callback = open_callback      # callable(QuiverGraph)

        # draggable node
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setZValue(0)

        # hover highlight: just blue border
        self.setAcceptHoverEvents(True)
        self._hovered = False
        self._hover_pen = QPen(QColor(0, 120, 255), 3.0, Qt.SolidLine, Qt.SquareCap, Qt.MiterJoin)



    def add_edge(self, e: 'EdgeItem'):
        self.edges.append(e)

    def itemChange(self, change, value):
        if change in (QGraphicsItem.ItemPositionChange, QGraphicsItem.ItemPositionHasChanged):
            for e in self.edges:
                e.update_path()
        return super().itemChange(change, value)

    # ----- hover handlers -----
    def hoverEnterEvent(self, event):
        self._hovered = True
        self.update()  # trigger repaint
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    # ----- double click -> pick component (if needed) -> confirm -> open -----
    def mouseDoubleClickEvent(self, event):
        if not self.sub_quivers:
            return
        
        # If all subquivers are isomorphic, just open the first one
        if len(self.sub_quivers) > 1:
            first_qg = self.sub_quivers[0]
            if all(is_isomorphic(first_qg, qg) for qg in self.sub_quivers[1:]):
                chosen_qg = first_qg
                resp = QMessageBox.question(
                    None,
                    "Open Quiver",
                    f"All components are isomorphic.\nOpen this quiver in a new calculation window?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if resp != QMessageBox.Yes:
                    return
                if callable(self.open_callback):
                    self.open_callback(chosen_qg)
                super().mouseDoubleClickEvent(event)
                return
            # Remove isomorphic duplicates from sub_quivers
            unique_quivers = []
            for qg in self.sub_quivers:
                if not any(is_isomorphic(qg, uq) for uq in unique_quivers):
                    unique_quivers.append(qg)
            
            self.sub_quivers = unique_quivers
            
        

        if len(self.sub_quivers) == 1:
            chosen_qg = self.sub_quivers[0]
        else:
            class LeafComponentsDialog(QDialog):
                """
                Modal picker that lists all sub-quivers in a leaf, each with:
                [Open] button + live preview + basic stats.
                Returns the chosen QuiverGraph via self.selected_quiver.
                """
                def __init__(self, sub_quivers: List[QuiverGraph], parent=None):
                    super().__init__(parent)
                    self.setWindowTitle(f"{len(sub_quivers)} Component{'s' if len(sub_quivers) > 1 else ''} in Leaf")
                    self.selected_quiver: Optional[QuiverGraph] = None

                    layout = QVBoxLayout(self)

                    scroll = QScrollArea()
                    scroll.setWidgetResizable(True)

                    container = QWidget()
                    vbox = QVBoxLayout(container)

                    for idx, Q in enumerate(sub_quivers, start=1):
                        group = QWidget()
                        group_layout = QHBoxLayout(group)

                        # Open button
                        open_btn = QPushButton("Open")
                        open_btn.clicked.connect(lambda _, q=Q: self._choose(q))
                        group_layout.addWidget(open_btn)

                        # Preview
                        view = QGraphicsView(_StaticScene(Q, parent=self))
                        view.setRenderHint(QPainter.Antialiasing)
                        view.setMinimumWidth(220)
                        view.setMinimumHeight(180)
                        group_layout.addWidget(view)

                        vbox.addWidget(group)

                    container.setLayout(vbox)
                    scroll.setWidget(container)
                    layout.addWidget(scroll)

                def _choose(self, q: QuiverGraph):
                    self.selected_quiver = q
                    self.accept()

            dlg = LeafComponentsDialog(self.sub_quivers, parent=self.scene().views()[0] if self.scene().views() else None)
            if dlg.exec() != QDialog.Accepted or dlg.selected_quiver is None:
                return
            chosen_qg = dlg.selected_quiver

        # confirm
        resp = QMessageBox.question(
            None,
            "Open Quiver",
            "Open this quiver in a new calculation window?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if resp != QMessageBox.Yes:
            return

        # open via callback
        if callable(self.open_callback):
            self.open_callback(chosen_qg)

        super().mouseDoubleClickEvent(event)

        
    
    # ----- custom paint to draw blue border on hover -----
    def paint(self, painter: QPainter, option, widget=None):
        # draw the pixmap first
        super().paint(painter, option, widget)

        if self._hovered:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(self._hover_pen)
            painter.setBrush(Qt.NoBrush)
            # draw *inside* the item rect to avoid clipping
            r = self.boundingRect().adjusted(1.5, 1.5, -1.5, -1.5)
            painter.drawRect(r)
    
    


class EdgeItem(QGraphicsPathItem):
    """
    Edge that renders `mult` parallel strands between src and dst, each with its own arrowhead.
    """
    def __init__(self, src: 'NodeItem', dst: 'NodeItem', multiplicity: int, stroke_w: float = 2.0):
        super().__init__()
        self.src = src
        self.dst = dst
        self.mult = max(1, int(multiplicity))
        self.stroke_w = float(stroke_w)

        pen = QPen(Qt.black, 2.0)
        pen.setCapStyle(Qt.FlatCap)
        pen.setJoinStyle(Qt.MiterJoin)
        self.setPen(pen)
        self.setZValue(2)

        # geometry params (scene units)
        self.arrow_len    = 12.0
        self.arrow_half_w = 6.0
        self.arrow_spread = 12.0   # perpendicular spacing between strands/heads

        # internal: list of strand endpoints for painting
        self._strands: list[tuple[QPointF, QPointF]] = []

        src.add_edge(self)
        dst.add_edge(self)
        self.update_path()

    # ---------- helpers ----------
    def _scene_rect(self, item: QGraphicsPixmapItem) -> QRectF:
        return item.mapRectToScene(item.boundingRect())

    def _shrink_rect(self, r: QRectF, stroke: float) -> QRectF:
        s = stroke / 2.0
        return r.adjusted(s, s, -s, -s)

    def _edge_segments(self, r: QRectF):
        return (
            (QPointF(r.left(),  r.top()),    QPointF(r.right(), r.top()   )),
            (QPointF(r.right(), r.top()),    QPointF(r.right(), r.bottom())),
            (QPointF(r.right(), r.bottom()), QPointF(r.left(),  r.bottom())),
            (QPointF(r.left(),  r.bottom()), QPointF(r.left(),  r.top()   )),
        )

    def _seg_intersection(self, p1, p2, q1, q2):
        x1,y1 = p1.x(), p1.y(); x2,y2 = p2.x(), p2.y()
        x3,y3 = q1.x(), q1.y(); x4,y4 = q2.x(), q2.y()
        den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(den) < 1e-12: return (False, QPointF(), 0.0)
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / den
        u = ((x1-x3)*(y1-y2) - (y1-y3)*(x1-x2)) / den
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return (True, QPointF(x1 + t*(x2-x1), y1 + t*(y2-y1)), t)
        return (False, QPointF(), 0.0)

    def _anchor_on_rect(self, p_from: QPointF, p_to: QPointF, rect: QRectF) -> QPointF:
        # pick intersection closest to p_to along the ray
        best = None; best_cost = 1e9
        for a, b in self._edge_segments(rect):
            ok, pt, t = self._seg_intersection(p_from, p_to, a, b)
            if ok:
                cost = abs(t - 1.0)
                if cost < best_cost:
                    best_cost = cost; best = pt
        return best if best is not None else rect.center()

    def _unit_perp_len(self, v: QPointF):
        L = (v.x()**2 + v.y()**2) ** 0.5
        if L == 0: return QPointF(0,0), QPointF(0,0), 0.0
        u = QPointF(v.x()/L, v.y()/L)
        n = QPointF(-u.y(), u.x())
        return u, n, L

    # ---------- geometry + path ----------
    def update_path(self):
        # raw anchors on the frames
        r1 = self._shrink_rect(self._scene_rect(self.src), self.stroke_w)
        r2 = self._shrink_rect(self._scene_rect(self.dst), self.stroke_w)
        c1, c2 = r1.center(), r2.center()
        a = self._anchor_on_rect(c1, c2, r1)
        b = self._anchor_on_rect(c1, c2, r2)

        u, n, L = self._unit_perp_len(b - a)
        if L == 0:
            self._strands = []
            self.setPath(QPainterPath())
            return

        # base shortened segment
        base_start = QPointF(a.x(), a.y())
        base_end   = QPointF(b.x(), b.y())

        # build parallel strands offset along n (centered set)
        self._strands = []
        center = (self.mult - 1) * 0.5
        comp = QPainterPath()
        for k in range(self.mult):
            off = (k - center) * self.arrow_spread
            s = QPointF(base_start.x() + n.x()*off, base_start.y() + n.y()*off)
            e = QPointF(base_end.x()   + n.x()*off, base_end.y()   + n.y()*off)
            self._strands.append((s, e))
            if k == 0:
                comp.moveTo(s)
            comp.lineTo(e)
        # the QGraphicsPathItem holds any path; we still paint strands individually for heads
        self.setPath(comp)
        self.prepareGeometryChange()
        self.update()

    # ---------- paint ----------
    def paint(self, painter: QPainter, option, widget=None):
        painter.setPen(self.pen())
        # draw each strand line
        for s, e in self._strands:
            painter.drawLine(s, e)

        # draw arrowhead on each strand
        for s, e in self._strands:
            u, n, L = self._unit_perp_len(e - s)
            if L <= 1e-6: 
                continue
            # tip a touch inset from end
            tip = QPointF(e.x(), e.y())
            base_center = QPointF(tip.x() - u.x()*self.arrow_len, tip.y() - u.y()*self.arrow_len)
            left  = QPointF(base_center.x() + n.x()*self.arrow_half_w,
                            base_center.y() + n.y()*self.arrow_half_w)
            right = QPointF(base_center.x() - n.x()*self.arrow_half_w,
                            base_center.y() - n.y()*self.arrow_half_w)
            tri = QPainterPath(tip); tri.lineTo(left); tri.lineTo(right); tri.closeSubpath()
            painter.setBrush(QBrush(Qt.black))
            painter.setPen(Qt.NoPen)
            painter.drawPath(tri)

# ─────────────────────────── DAG layout ───────────────────────────

def layered_layout(hasse_mult: np.ndarray, *, xgap: int = 40, ygap: int = 87) -> Dict[int, QPointF]:
    """
    Simple longest-path layering for a DAG adjacency (hasse_mult > 0).
    Returns {node_index -> QPointF(x, y)} for node centers.
    """
    A = (hasse_mult > 0).astype(int)
    n = A.shape[0]

    # Kahn topo order
    indeg = A.sum(axis=0).tolist()
    order: List[int] = []
    queue: List[int] = [i for i in range(n) if indeg[i] == 0]
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in range(n):
            if A[u, v]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)

    # longest distance from any source
    layer = [0] * n
    for u in order:
        for v in range(n):
            if A[u, v]:
                layer[v] = max(layer[v], layer[u] + 1)

    # group by layer
    buckets: Dict[int, List[int]] = {}
    for i, ell in enumerate(layer):
        buckets.setdefault(ell, []).append(i)

    # center nodes per layer
    pos: Dict[int, QPointF] = {}
    approx_node_w = 380  # rough width used to center per layer
    for ell, nodes in buckets.items():
        width = len(nodes)
        for j, idx in enumerate(nodes):
            x = (j - (width - 1) / 2.0) * (approx_node_w + xgap)
            y = ell * ygap
            pos[idx] = QPointF(x, y)
    return pos


# ─────────────────────────── main view ───────────────────────────

class HasseDiagramView(QGraphicsView):
    """
    Draggable Hasse diagram:
      • each node is a leaf thumbnail (black box with components & ⊗)
      • edges/arrows float ABOVE nodes and track during drags
      • layout avoids overlap initially; user can rearrange freely
    """
    def __init__(self, full_qg, M, all_leaves, hasse_mult, idx2node, parent=None, open_quiver_callback=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(Qt.white)

        scene = QGraphicsScene(self)
        scene.setBackgroundBrush(Qt.white)
        self.setScene(scene)

        self._panning = False
        self._pan_start = None
        self._user_scaled = False          # <- track if user has zoomed
        self._initial_fit_done = False     # <- run one-time fit
        self.viewport().setCursor(Qt.OpenHandCursor)

        # 1) base positions from the full graph (IDs preserved)
        base_pos = compute_base_pos(full_qg)

        # 2) pre-render pixmaps for each leaf (pass idx2node)
        layers = self._layers_from_hasse(hasse_mult)
        leaf_payloads = [
            render_leaf_thumbnail_and_subquivers(
                full_qg, leaf, base_pos,
                idx2node=idx2node,
                pad=16, box_pad=10, box_pen_w=2, supersample=2, bg=Qt.white
            )
            for leaf in all_leaves
        ]
        node_pixmaps = [pm for (pm, _subs) in leaf_payloads]


        # 3) initial non-overlapping positions by per-layer packing
        xgap_min, ygap_min = -30, -30
        pos = {}
        y_cursor = 0.0
        for layer_indices in layers:
            widths  = [node_pixmaps[i].width()  for i in layer_indices]
            heights = [node_pixmaps[i].height() for i in layer_indices]
            if not widths:
                continue
            layer_h = max(heights)/1.25
            total_w = sum(widths)/1.25 
            x_cursor = - total_w / 2.0
            for idx, w in zip(layer_indices, widths):
                h = node_pixmaps[idx].height()
                pos[idx] = QPointF(x_cursor + w/2.0, y_cursor + layer_h/2.0)
                x_cursor += w + xgap_min
            y_cursor += layer_h + ygap_min

        # 4) add draggable NodeItems
        nodes: list[NodeItem] = []
        for i, (pm, sub_qs) in enumerate(leaf_payloads):
            item = NodeItem(pm, sub_quivers=sub_qs, open_callback=open_quiver_callback)
            cx, cy = pos[i].x(), pos[i].y()
            item.setPos(cx - pm.width()/2.0, cy - pm.height()/2.0)
            self.scene().addItem(item)
            nodes.append(item)

        # 5) add EdgeItems (on top); they auto-update as nodes move
        stroke_w = 2.0  # must match box_pen_w used when drawing the black frame inside thumbnails
        n = len(all_leaves)
        for i in range(n):
            for j in range(n):
                mult = int(hasse_mult[i, j])
                if mult <= 0:
                    continue
                edge = EdgeItem(nodes[i], nodes[j], multiplicity=mult, stroke_w=stroke_w)
                scene.addItem(edge)

        rect = scene.itemsBoundingRect()
        pad = max(rect.width(), rect.height())  # scale with diagram size
        scene.setSceneRect(rect.adjusted(-pad, -pad, pad, pad))
        # Do first fit after the widget is shown (so it knows its real size)
        QTimer.singleShot(0, self._fit_all)
        # enable item dragging; keep view itself non-dragging (so mouse drags items)
        self.setDragMode(QGraphicsView.NoDrag)


    def _fit_all(self, margin: float = 20.0):
        if self.scene() is None:
            return
        rect = self.scene().itemsBoundingRect().marginsAdded(QMarginsF(margin, margin, margin, margin))
        if rect.isNull() or not rect.isValid():
            return
        # Reset any prior zoom/pan transform before fitting
        self.setTransform(QTransform())
        self.fitInView(rect, Qt.KeepAspectRatio)
        self._initial_fit_done = True
        self._user_scaled = False  # just fitted

    def resizeEvent(self, e):
        super().resizeEvent(e)
        # Keep it fitted on window resizes until the user manually zooms
        if not self._user_scaled and self._initial_fit_done:
            self._fit_all()

    def wheelEvent(self, e):
        factor = 1.15 if e.angleDelta().y() > 0 else 1/1.15
        self.scale(factor, factor)
        self._user_scaled = True  # user took control

    def keyPressEvent(self, e):
        # Press 'R' to refit everything any time
        if e.key() in (Qt.Key_R):
            self._fit_all()
            e.accept()
            return
        super().keyPressEvent(e)


    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            item = self.itemAt(e.pos())
            if item is None:
                self._panning = True
                self._pan_start = e.pos()
                self.viewport().setCursor(Qt.ClosedHandCursor)
                e.accept()
                return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        # use getattr to be safe even if __init__ missed init
        if getattr(self, "_panning", False) and getattr(self, "_pan_start", None) is not None:
            delta = e.pos() - self._pan_start
            self._pan_start = e.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if getattr(self, "_panning", False) and e.button() == Qt.LeftButton:
            self._panning = False
            self._pan_start = None
            self.viewport().setCursor(Qt.OpenHandCursor)
            e.accept()
            return
        super().mouseReleaseEvent(e)


    def wheelEvent(self, e):
        factor = 1.15 if e.angleDelta().y() > 0 else 1/1.15
        self.scale(factor, factor)

    # — layers helper (unchanged) —
    def _layers_from_hasse(self, hasse_mult) -> list[list[int]]:
        A = (hasse_mult > 0).astype(int)
        n = A.shape[0]
        indeg = A.sum(axis=0).tolist()
        order, queue = [], [i for i in range(n) if indeg[i] == 0]
        while queue:
            u = queue.pop(0); order.append(u)
            for v in range(n):
                if A[u, v]:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        queue.append(v)
        layer = [0]*n
        for u in order:
            for v in range(n):
                if A[u, v]:
                    layer[v] = max(layer[v], layer[u] + 1)
        buckets = {}
        for i, ell in enumerate(layer):
            buckets.setdefault(ell, []).append(i)
        return [buckets[k] for k in sorted(buckets.keys())]

    def _add_arrowhead(self, scene: QGraphicsScene, p1: QPointF, p2: QPointF, *, scale: float = 12.0, count: int = 1, z: float = 1.0):
        v = p2 - p1
        L = (v.x()**2 + v.y()**2)**0.5
        if L == 0:
            return
        ux, uy = v.x()/L, v.y()/L
        px, py = -uy, ux
        for k in range(count):
            shift = (k - (count-1)/2.0) * 7.0
            tip  = p2 + QPointF(px*shift, py*shift)
            base = tip - QPointF(ux*scale, uy*scale)
            left = base + QPointF(px*scale*0.6, py*scale*0.6)
            right= base - QPointF(px*scale*0.6, py*scale*0.6)

            tri = QPainterPath(tip)
            tri.lineTo(left); tri.lineTo(right); tri.closeSubpath()

            head = QGraphicsPathItem(tri)
            head.setBrush(QBrush(Qt.black))
            head.setPen(Qt.NoPen)
            head.setZValue(z)
            scene.addItem(head)





def _edge_points(r: QRectF) -> List[Tuple[QPointF, QPointF]]:
    return [
        (QPointF(r.left(),  r.top()),    QPointF(r.right(), r.top()   )),  # top
        (QPointF(r.right(), r.top()),    QPointF(r.right(), r.bottom())),  # right
        (QPointF(r.right(), r.bottom()), QPointF(r.left(),  r.bottom())),  # bottom
        (QPointF(r.left(),  r.bottom()), QPointF(r.left(),  r.top()   )),  # left
    ]

def _seg_intersection(p1: QPointF, p2: QPointF, q1: QPointF, q2: QPointF):
    # returns (hit, point, t_on_p)   with t in [0,1] along p1->p2
    x1,y1 = p1.x(), p1.y(); x2,y2 = p2.x(), p2.y()
    x3,y3 = q1.x(), q1.y(); x4,y4 = q2.x(), q2.y()
    den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(den) < 1e-12:
        return (False, QPointF(), 0.0)
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / den
    u = ((x1-x3)*(y1-y2) - (y1-y3)*(x1-x2)) / den
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return (True, QPointF(x1 + t*(x2-x1), y1 + t*(y2-y1)), t)
    return (False, QPointF(), 0.0)



def _apply_leaf_ranks_inplace(qg_sub: QuiverGraph, v: np.ndarray, idx2node: list):
    """
    Update node attributes in qg_sub to reflect the ranks in v (matrix index space).
    Assumes qg_sub nodes are original IDs (induced subgraph from full_qg).
    """
    for i, r in enumerate(v):
        if r <= 0:
            continue
        node_id = idx2node[i]
        if node_id in qg_sub.nodes:
            qg_sub.nodes[node_id]["gp_rank"] = int(r)

            # If your renderer uses a string label, update or regenerate it too.
            # Adjust this to match whatever _StaticScene expects.
            if "label" in qg_sub.nodes[node_id]:
                qg_sub.nodes[node_id]["label"] = f"U({int(r)})"
            # Optionally clear other label-ish fields so _StaticScene recomputes from gp_rank:
            for k in ("latex", "name", "text", "title"):
                qg_sub.nodes[node_id].pop(k, None)


