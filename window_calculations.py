from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QCheckBox, QGraphicsView, QGraphicsScene,
                               QMessageBox, QDialog, QTextEdit,
                               QInputDialog)
from PySide6.QtGui    import QPen, QBrush, QColor, QPainter
from PySide6.QtCore   import QRectF, Qt, QLineF

from constants import NODE_RADIUS, PEN_NODE, SCENE_WIDTH, SCENE_HEIGHT
from graph_model import QuiverGraph, compute_balance
from nx_layouts import plot_caterpillar, plot_sunshine
from nx_dynkin import Dynkin_A, Dynkin_D, Dynkin_E
from Qt_custom_boxes import show_scrollable_message, show_warning_with_link
from monopole_formula import hilbert_series_from_quiver_graph


import math
import sympy as sp
import networkx as nx
import copy
from collections import Counter
import time

class _StaticScene(QGraphicsScene):
    """Draws the graph once; mouse & key events are ignored."""

    def __init__(self, g: QuiverGraph, *args, **kw):
        super().__init__(*args, **kw)
        self._draw_graph(g)
        self.setSceneRect(self.itemsBoundingRect())

    # suppress interaction
    def mousePressEvent(self, _):   pass
    def mouseMoveEvent (self, _):   pass
    def mouseReleaseEvent(self, _): pass
    def keyPressEvent(self,  _):    pass

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
            # ---------- 1. node positions ----------
            cycles = nx.cycle_basis(nx.Graph(g))
            if not cycles:
                pos = plot_caterpillar(g,
                    hsep=5 * NODE_RADIUS + 10,  # horizontal spacing
                    vsep=5 * NODE_RADIUS + 10,  # vertical spacing
                )
            elif len(cycles) == 1:
                pos = plot_sunshine(g,
                    radius=5 * NODE_RADIUS + 10,  # radius of the cycle
                    vsep=5 * NODE_RADIUS + 10,    # vertical spacing for branches
                )
            else:
                pos = nx.spring_layout(g, seed=42, scale=5 * NODE_RADIUS + 10)

                5 * NODE_RADIUS + 10

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
                flav_gauge = data.get("flav_gauge", "gauge")

                # shape
                shape_func = self.addEllipse if flav_gauge == "gauge" else self.addRect
                item = shape_func(
                    QRectF(x - NODE_RADIUS, y - NODE_RADIUS,
                        2 * NODE_RADIUS, 2 * NODE_RADIUS),
                    PEN_NODE, QBrush(Qt.white)
                )
                item.setData(0, n)

                # label text identical to editor rule
                gp_type = data.get("gp_type", "?")
                gp_rank = data.get("gp_rank", "?")
                if gp_type == "U" and flav_gauge == "flav":
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
        """
        Return a plain-text tooltip string: group + optional balance.
        """
        data = g.nodes[node_id]
        gp_type   = data.get("gp_type", "?")
        gp_rank   = data.get("gp_rank", "?")
        flav_flag = data.get("flav_gauge", "")

        gp_name = f"{gp_type}({gp_rank})\nType: {flav_flag}"

        balance = compute_balance(g, node_id)

        output_str = f"Group: {gp_name}"
        if balance is not None:
            output_str += f"\nBalance: b = {balance}"
        return output_str




class CalculationsWindow(QMainWindow):
    def __init__(self, g: QuiverGraph, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Quiver Calculations")
        self.graph = g                # reference only; NOT copied

        # --- central, read-only view ---
        self.scene = _StaticScene(self.graph)
        self.view  = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.resize(1000, 650)

        # --- side controls ---
        
        side = QWidget()
        s_lay = QVBoxLayout(side)
        s_lay.setAlignment(Qt.AlignTop)        

        # Side-by-side Gauge/Ungauge buttons
        self.gauge_btn = QPushButton("Gauge")
        self.gauge_btn.clicked.connect(self._gauge)
        self.ungauge_btn = QPushButton("Ungauge")
        self.ungauge_btn.clicked.connect(self._ungauge)
        gauge_ungauge_row = QWidget()
        gauge_ungauge_layout = QHBoxLayout(gauge_ungauge_row)
        gauge_ungauge_layout.setContentsMargins(0, 0, 0, 0)
        gauge_ungauge_layout.addWidget(self.gauge_btn)
        gauge_ungauge_layout.addWidget(self.ungauge_btn)
        # Enable/disable buttons based on presence of flavour nodes
        flavour_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("flav_gauge") == "flav"]
        if bool(flavour_nodes): # there are flavour nodes
            self.gauge_btn.setEnabled(True)
            self.gauge_btn.setToolTip("Gauge quiver by connecting all (unitary) flavour nodes into a U(1) gauge node.")
            self.ungauge_btn.setEnabled(False)
            self.gauge_btn.setToolTip("Ungauging is only valid for unframed/flavourless quivers.")
        elif not flavour_nodes: # no flavour nodes (flavourless/unframed)
            self.gauge_btn.setEnabled(False)
            self.gauge_btn.setToolTip("Gauging is only valid for framed quivers (quivers with flavour nodes).")
            self.ungauge_btn.setEnabled(True)
            self.ungauge_btn.setToolTip("Ungauge/decouple a U(1) factor, analogous to fixing centre of mass.")
            

        self.combine_btn = QPushButton("Combine Flavours")
        self.combine_btn.setToolTip("If multiple unitary flavour nodes are connected to the same gauge node,"
                                              "combine them into one flavour node whose rank is sum of originals.")
        self.combine_btn.clicked.connect(self._combine_flavours)

        self.color_chk = QCheckBox("Color by balance")
        self.color_chk.setToolTip("red = underbalanced\n"
                                  "white = balanced\n"
                                  "gray = overbalanced")
        self.color_chk.setChecked(True)
        self._toggle_balance_colours(2)
        self.color_chk.stateChanged.connect(self._toggle_balance_colours)


        self.glob_symm_subgp_C_btn = QPushButton("Coulomb Global Symmetry Subgroup")
        self.glob_symm_subgp_C_btn.setToolTip("Calculate Coulomb global symmetry subgroup.\n"
                                              "(this may or may not be the global symmetry group)\n"
                                              "Currently only developed for U and SU gauge nodes.")
        self.glob_symm_subgp_C_btn.clicked.connect(self._glob_symm_subgp_C)

        self.glob_symm_subgp_H_btn = QPushButton("Higgs Global Symmetry Subgroup")
        self.glob_symm_subgp_H_btn.setToolTip("Calculate Higgs global symmetry subgroup.\n"
                                              "(this may or may not be the global symmetry group)\n"
                                              "Currently only developed for unitary flavour nodes.")
        self.glob_symm_subgp_H_btn.clicked.connect(self._glob_symm_subgp_H)

        self.hilbert_series_btn = QPushButton("Coulomb Branch Hilbert Series")
        self.hilbert_series_btn.setToolTip("Calculate the Hilbert Series of the Coulomb branch for the current quiver.")
        self.hilbert_series_btn.clicked.connect(self._HS_C)

        
        s_lay.addWidget(gauge_ungauge_row)
        s_lay.addWidget(self.combine_btn)
        s_lay.addWidget(self.color_chk)
        
        s_lay.addWidget(self.glob_symm_subgp_C_btn)
        s_lay.addWidget(self.glob_symm_subgp_H_btn)

        s_lay.addWidget(self.hilbert_series_btn)

        # --- main layout ---
        central = QWidget(); lay = QHBoxLayout(central)
        lay.addWidget(self.view, 4); lay.addWidget(side, 1)
        self.setCentralWidget(central); self.resize(800, 500)


    def _combine_flavours(self):
        """
        Create a **new** CalculationsWindow in which any gauge node that had
        multiple connected U-flavour nodes is now connected to exactly one
        U-flavour node whose rank is the sum of the originals.

        The current window is closed (its scene is cleared in closeEvent),
        so the original graph view is discarded.
        """
        new_g = copy.deepcopy(self.graph)           # work on a fresh copy

        # Generate a fresh node id counter
        next_id = max(new_g.nodes, default=-1) + 1

        for gauge in list(new_g.nodes):          # snapshot of original IDs
            if gauge not in new_g:               
                continue      
            ndata = new_g.nodes[gauge]
            if ndata.get("flav_gauge") != "gauge":
                continue  # skip non-gauge nodes

            # All directly-connected U-flavour neighbours
            u_flavours = [
                n for n in new_g.neighbors(gauge)
                if new_g.nodes[n].get("flav_gauge") == "flav"
                and new_g.nodes[n].get("gp_type") == "U"
            ]

            if len(u_flavours) <= 1:
                continue  # nothing to combine for this gauge

            # ----- 1. compute combined attributes -----
            total_rank = sum(new_g.nodes[f]["gp_rank"] for f in u_flavours)
            # place new node at average position of its constituents
            avg_x = sum(new_g.nodes[f]["pos"][0] for f in u_flavours) / len(u_flavours)
            avg_y = sum(new_g.nodes[f]["pos"][1] for f in u_flavours) / len(u_flavours)

            # ----- 2. add the combined flavour node -----
            new_g.add_node(
                next_id,
                gp_type="U",
                gp_rank=total_rank,
                flav_gauge="flav",
                pos=(avg_x, avg_y)
            )
            new_g.add_edge(next_id, gauge)

            # ----- 3. remove old flavour nodes (+ implicit edges) -----
            for old in u_flavours:
                new_g.remove_node(old)

            next_id += 1

        # If nothing was combined, tell the user and keep the current window
        if new_g.number_of_nodes() == self.graph.number_of_nodes():
            show_scrollable_message(
                self, "No change",
                "There were no gauge nodes with multiple connected U-flavours."
            )
            return

        # ----- 4. launch a fresh CalculationsWindow and hand it to MainWindow -----
        main_win = self.parent()              # MainWindow created this window
        new_win  = CalculationsWindow(new_g, parent=main_win)
        main_win._calc_win = new_win          # keep a Python reference
        new_win.show()

        # close—and thereby delete—the current window and its graph
        self.close()

    def _toggle_balance_colours(self, state: int):
        """Re-colour nodes according to balance when box is ticked."""
        for item in self.scene.items():
            nid = item.data(0)
            if nid is None:                       # skip edges / labels
                continue
            bal = compute_balance(self.graph, nid)
            if state and bal is not None:         # colouring ON
                if bal < 0:
                    colour = QColor("#ff9898")    # light red
                elif bal > 0:
                    colour = QColor("#bebdbd")    # light grey
                else:
                    colour = Qt.white
            else:                                 # colouring OFF
                colour = Qt.white
            item.setBrush(QBrush(colour))



    def _glob_symm_subgp_H(self):
        gps_dim = -1

        # SU from higher multiplicity edges
        edge_mults = []
        for u, v in self.graph.edges():
            edge_mult = self.graph.number_of_edges(u, v)
            if edge_mult > 1:
                # only consider edges with multiplicity > 1, since SU(1) is trivial
                edge_mults.append(edge_mult)
        glob_symm_subgp_str_e = " × ".join(
            f"SU({n})^{m}" if m > 1 else f"SU({n})"
            for n, m in sorted(Counter(edge_mults).items(), reverse=True))
        for mult in edge_mults:
            gps_dim += mult**2 - 1

        # SU from flav nodes
        flavs = []
        for n, d in self.graph.nodes(data=True):
            if d.get("flav_gauge") == "flav":
                gp_type = d.get("gp_type", "?")
                gp_rank = d.get("gp_rank", "?")
                if gp_type == "U":
                    flavs.append(gp_rank)
                    gps_dim += gp_rank**2
                else:
                    QMessageBox.warning(self, "No Subgroup Found", "Currently, Higgs global symmetry subgroup calculation is only implemented for unitary flavour nodes.")
                    return
                    
        flavs = sorted(flavs, reverse=True)

        if not flavs:
            QMessageBox.warning(self, "No Subgroup Found", "No flavour nodes found.")
            return
        else:
            counts = Counter(flavs)
            glob_symm_subgp_str_f = " × ".join(
                f"U({n})^{m}" if m > 1 else f"U({n})"
                for n, m in sorted(counts.items(), reverse=True))
            
            
            
            if 1 in flavs:
                glob_symm_subgp_str_f_rem_U1 = " × ".join(f"U({n})^{m}" if m > 1 else f"U({n})" for n, m in sorted(Counter(flavs[:-1]).items(), reverse=True))
                if not glob_symm_subgp_str_f_rem_U1:
                    if edge_mults == []:
                        glob_symm_subgp_str_f_rem_U1 = r"trivial group"
                    else:
                        glob_symm_subgp_str_f_rem_U1 = glob_symm_subgp_str_f_rem_U1[:-3]  # remove trailing " × " 
                output_str = (
                    f"{glob_symm_subgp_str_e} × S(  {glob_symm_subgp_str_f}  )\n"
                    f"≅ {glob_symm_subgp_str_e} × {glob_symm_subgp_str_f} / U(1)\n"
                    f"≅ {glob_symm_subgp_str_e} × {glob_symm_subgp_str_f_rem_U1}\n\n"
                    f"Dimension: {gps_dim}"
                )

            else:
                output_str = (
                    f"{glob_symm_subgp_str_e} × S(  {glob_symm_subgp_str_f}  )\n"
                    f"≅ {glob_symm_subgp_str_e} × {glob_symm_subgp_str_f} / U(1)\n\n"
                    f"Dimension: {gps_dim}"
                )

            show_scrollable_message(self, "Higgs Global Symmetry Subgroup", output_str)
            return




    def _glob_symm_subgp_C(self):

        gps = []

        G_removed_overb = copy.deepcopy(self.graph)
    
        for n, d in self.graph.nodes(data=True):
            if d.get("flav_gauge") == "gauge":
                gp_type = d.get("gp_type", "?")
                gp_rank = d.get("gp_rank", "?")
                gp_bal = compute_balance(self.graph, n)
                if gp_type not in ("U", "SU"):
                    QMessageBox.warning(self, "No Subgroup Found", "Currently, Coulomb global symmetry subgroup calculation is only implemented for U/SU gauge nodes.")
                    return
                elif gp_bal < 0:
                    QMessageBox.warning(self, "No Subgroup Found", "Currently, Coulomb global symmetry subgroup calculation is not supported for underbalanced gauge groups.")
                    return
                elif gp_type == "SU" and gp_bal == 0:
                    show_warning_with_link(self, "No Subgroup Found",
                                           r"Currently, Coulomb global symmetry subgroup calculation is not supported for balanced SU gauge groups." \
                                           "See Appendix D of <a href=https://doi.org/10.1007/JHEP03(2021)242>https://doi.org/10.1007/JHEP03(2021)242</a> for details on such calculations.")
                    return
                elif gp_type == "U" and gp_bal > 0:
                    gps.append(("U", 1))
                    G_removed_overb.remove_node(n)
                elif gp_type == "SU" and gp_bal > 0:
                    G_removed_overb.remove_node(n)
            elif d.get("flav_gauge") == "flav":
                    G_removed_overb.remove_node(n)  

        G_removed_overb = nx.Graph(G_removed_overb) # not multigraph

        for component_nodes in nx.connected_components(G_removed_overb):
            subg = G_removed_overb.subgraph(component_nodes).copy()

            subg_size = subg.number_of_nodes()

            if nx.is_isomorphic(subg, Dynkin_A(subg_size)): # linear, ie Dynkin diagram A_n
                gps.append(("SU", subg_size+1))
            elif subg_size >= 4 and nx.is_isomorphic(subg, Dynkin_D(subg_size)): # Dynkin diagram D_n
                gps.append(("SO", 2*subg_size))
            elif subg_size in (6, 7, 8) and nx.is_isomorphic(subg, Dynkin_E(subg_size)): # Dynkin diagram E_n
                gps.append(("E", subg_size))
            else:
                QMessageBox.warning(self, "No Subgroup Found", "A connected component of balanced U gauge nodes does not match any known Dynkin diagram.")

        # SU from higher multiplicity edges
        edge_mults = []
        for u, v in self.graph.edges():
            edge_mult = self.graph.number_of_edges(u, v)
            edge_mults.append(edge_mult)

        priority = {"E": 0, "SO": 1, "SU": 2, "U": 3}
        gps.sort(key=lambda x: (priority.get(x[0], 99), -x[1]))

        gps_str = []
        gps_dim = 0
        for gp in gps:
            gp_type = gp[0]
            gp_rank = gp[1]
            if gp_type == "U":
                gps_str.append(f"U({gp_rank})")
                gps_dim += gp_rank**2
            elif gp_type == "SU":
                gps_str.append(f"SU({gp_rank})")
                gps_dim += gp_rank**2 - 1
            elif gp_type == "SO":
                gps_str.append(f"SO({gp_rank})")
                gps_dim += gp_rank * (gp_rank - 1) / 2
            elif gp_type == "E":
                gps_str.append(f"E_{gp_rank}")
                if gp_rank == 6:
                    gps_dim += 78
                elif gp_rank == 7:
                    gps_dim += 133
                elif gp_rank == 8:
                    gps_dim += 248
        
        counts = Counter(gps_str)
        seen = set()
        gps_str_combine = " × ".join(
            f"{g}^{counts[g]}" if counts[g] > 1 else g
            for g in gps_str if not (g in seen or seen.add(g))
        )
        
        output_str = (
                    f"{gps_str_combine}\n\n"
                    f"Dimension: {gps_dim}"
                )
        
        show_scrollable_message(self, "Coulomb Global Symmetry Subgroup", output_str)


    def _gauge(self):
        """
        Convert all U-flavour nodes to gauge nodes.
        Only allowed if there are U-flavour nodes and no non-U flavour nodes.
        """

        new_g = copy.deepcopy(self.graph)  # work on a fresh copy

        # Find all flavour nodes
        flavour_nodes = [n for n, d in new_g.nodes(data=True) if d.get("flav_gauge") == "flav"]
        if not flavour_nodes:
            QMessageBox.warning(self, "No Flavour Nodes", "There are no flavour nodes to gauge.")
            return

        # Check all flavour nodes are U
        non_u_flavours = [n for n in flavour_nodes if new_g.nodes[n].get("gp_type") != "U"]
        if non_u_flavours:
            QMessageBox.warning(self, "Non-Unitary Flavour Nodes", "Currently, gauging is not supported for non-unitary flavour nodes.")
            return

        # ----- 1. create a new U(1) gauge node -----
        new_U1 = max(new_g.nodes, default=-1) + 1
        new_g.add_node(new_U1, gp_type="U", flav_gauge="gauge")

        # ----- connect all previous flavour nodes to the new U(1) gauge node -----
        # Edge multiplicity is rank of corresponding previous flavour node
        for flav in flavour_nodes:
            neighbour = next(new_g.neighbors(flav)) # Each flavour node has only one neighbour (the gauge node it's attached to)
            flav_rank = new_g.nodes[flav].get("gp_rank")
            for _ in range(flav_rank):
                new_g.add_edge(new_U1, neighbour)

        # ----- remove flav_gauge flag from previous flavour nodes -----
        for flav in flavour_nodes:
            new_g.remove_node(flav)

        # ----- launch a fresh CalculationsWindow and hand it to MainWindow -----
        main_win = self.parent()              # MainWindow created this window
        new_win  = CalculationsWindow(new_g, parent=main_win)
        main_win._calc_win = new_win          # keep a Python reference
        new_win.show()
        self.close()
        
        

    def _ungauge(self):
        # Check if there are any flavour nodes
        flavour_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("flav_gauge") == "flav"]
        if bool(flavour_nodes):  # unframed/flavourless quiver
            QMessageBox.warning(self, "Unframed Quiver", "Ungauging is only valid for unframed/flavourless quivers.")
            return

        # Check if there is a U(1) gauge node
        u1_gauge_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("flav_gauge") == "gauge" and d.get("gp_type") == "U" and d.get("gp_rank") == 1]
        if not u1_gauge_nodes:  # no U(1) gauge node
            QMessageBox.warning(self, "No U(1) Gauge Node",
                                "Currently, ungauging is only supported for U(1) gauge nodes." \
                                "To ungauge a higher-rank unitary gauge node, simply replace the circle with a squircle.")

        # Highlight U(1) gauge nodes on hover and allow selection by click
        class U1SelectorScene(_StaticScene):
            def __init__(self, g, u1_nodes, parent_win):
                super().__init__(g)
                self.u1_nodes = set(u1_nodes)
                self.parent_win = parent_win
                self.selected = None

            def mouseMoveEvent(self, event):
                pt = event.scenePos()
                hovered = None
                for item in self.items(pt):
                    nid = item.data(0)
                    if nid in self.u1_nodes:
                        hovered = item
                        break
                for item in self.items():
                    nid = item.data(0)
                    if nid in self.u1_nodes:
                        item.setBrush(QBrush(QColor("#90daff")) if item is hovered else QBrush(Qt.white))
                super().mouseMoveEvent(event)

            def mousePressEvent(self, event):
                pt = event.scenePos()
                for item in self.items(pt):
                    nid = item.data(0)
                    if nid in self.u1_nodes:
                        self.selected = nid
                        self.ungauge_selected()
                        return
                super().mousePressEvent(event)

            def ungauge_selected(self):
                nid = self.selected
                if nid is None:
                    return
                
                # Check if removing this U(1) gauge node would disconnect the graph
                temp_g = copy.deepcopy(self.parent_win.graph)
                temp_g.remove_node(nid)
                if not nx.is_connected(nx.Graph(temp_g)):
                    QMessageBox.warning(self.parent_win, "Disconnected Graph", "Ungauging on this U(1) gauge node results in a disconnected graph.")
                    return
                
                # Remove the U(1) gauge node and attach its neighbours as flavour nodes
                new_g = copy.deepcopy(self.parent_win.graph)
                neighbours = list(new_g.neighbors(nid))
                for n in neighbours:
                    # Add a new flavour node for each neighbour
                    flav_id = max(new_g.nodes, default=-1) + 1
                    edge_mult = new_g.number_of_edges(nid, n)
                    new_g.add_node(
                        flav_id,
                        gp_type="U",
                        gp_rank=edge_mult,
                        flav_gauge="flav"
                    )
                    new_g.add_edge(flav_id, n)
                new_g.remove_node(nid)
                main_win = self.parent_win.parent()
                new_win = CalculationsWindow(new_g, parent=main_win)
                main_win._calc_win = new_win
                new_win.show()

                self.parent_win.close()

        # Swap out the scene for selection
        self.view.setScene(U1SelectorScene(self.graph, u1_gauge_nodes, self))
        QMessageBox.information(self, "Select U(1) Gauge Node",
                    "Click a U(1) gauge node to ungauge it.\n"
                    "Valid nodes are highlighted on hover.")
    
    

    def _HS_C(self):
        gg, ok = QInputDialog.getInt(
            self, "Hilbert Series Cutoff",
            "Enter cutoff for Hilbert series computation.\nSeries will be calculated up to order double of cutoff selected.\n"
            "(WARNING: Computation time scales exponentially with calculation order, number of nodes and group rank.)",
            value=5, minValue=1, maxValue=20
            )
        if not ok:
            return
        
        # print(self.graph)
        # print(self.graph.edges(data=True))
        # print(self.graph.nodes(data=True))

        start_time = time.time()
        HS = hilbert_series_from_quiver_graph(self.graph, cutoff=gg)
        elapsed = time.time() - start_time

        HS_str = str(HS).replace("**", "^")

        output_str = HS_str
        show_scrollable_message(self,
                                f"Hilbert Series of Coulomb branch (Calculation time: {elapsed:.2f} secs)",
                                output_str
                                )
        return


    @staticmethod
    def _not_implemented():
        print("[TODO] button clicked – implement algorithm here")

    

