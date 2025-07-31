from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QCheckBox, QGraphicsView, QGraphicsScene,
                               QMessageBox, QDialog, QTextEdit)
from PySide6.QtGui    import QPen, QBrush, QColor, QPainter
from PySide6.QtCore   import QRectF, Qt, QLineF

from constants import NODE_RADIUS, PEN_NODE, SCENE_WIDTH, SCENE_HEIGHT
from graph_model import QuiverGraph
from nx_layouts import plot_caterpillar, plot_sunshine
from nx_dynkin import Dynkin_A, Dynkin_D, Dynkin_E
from Qt_custom_boxes import show_scrollable_message, show_warning_with_link


import math
import networkx as nx
import copy
from collections import Counter

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
            else:
                pos = plot_sunshine(g,
                    radius=5 * NODE_RADIUS + 10,  # radius of the cycle
                    vsep=5 * NODE_RADIUS + 10,    # vertical spacing for branches
                )

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
                if gp_type == "U" and flav_gauge == "flavour":
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

        gp_name = f"{gp_type}({gp_rank})"

        balance = compute_balance(g, node_id)
        if balance is not None:
            return f"Group: {gp_name}\nBalance = {balance}"
        else:
            return f"Group: {gp_name}"




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
        self.gauge_btn   = QPushButton("Gauge / Ungauge [tbd]")
        self.gauge_btn.clicked.connect(self._not_implemented)

        self.combine_btn = QPushButton("Combine Flavours")
        self.combine_btn.setToolTip("If multiple U/SU flavour nodes are connected to the same gauge node,"
                                              "combine them into one flavour node whose rank is sum of originals.")
        self.combine_btn.clicked.connect(self._combine_flavours)

        self.color_chk = QCheckBox("Color by balance")
        self.color_chk.setToolTip("red = underbalanced\n"
                                  "white = balanced\n"
                                  "gray = overbalanced")
        self.color_chk.stateChanged.connect(self._toggle_balance_colours)


        self.glob_symm_subgp_C_btn = QPushButton("Coulomb Global Symmetry Subgroup")
        self.glob_symm_subgp_C_btn.setToolTip("Calculate Coulomb global symmetry subgroup.\n"
                                              "(this may or may not be the global symmetry group)\n"
                                              "Currently only developed for U and SU gauge nodes.")
        self.glob_symm_subgp_C_btn.clicked.connect(self._glob_symm_subgp_C)

        self.glob_symm_subgp_H_btn = QPushButton("Higgs Global Symmetry Subgroup")
        self.glob_symm_subgp_H_btn.setToolTip("Calculate Higgs global symmetry subgroup.\n"
                                              "(this may or may not be the global symmetry group)\n"
                                              "Currently only developed for U/SU flavour nodes.")
        self.glob_symm_subgp_H_btn.clicked.connect(self._glob_symm_subgp_H)


        side = QWidget()
        s_lay = QVBoxLayout(side)
        s_lay.setAlignment(Qt.AlignTop)
        
        s_lay.addWidget(self.gauge_btn)
        s_lay.addWidget(self.combine_btn)
        s_lay.addWidget(self.color_chk)
        
        s_lay.addWidget(self.glob_symm_subgp_C_btn)
        s_lay.addWidget(self.glob_symm_subgp_H_btn)

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
                if new_g.nodes[n].get("flav_gauge") == "flavour"
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
                flav_gauge="flavour",
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
            if d.get("flav_gauge") == "flavour":
                gp_type = d.get("gp_type", "?")
                gp_rank = d.get("gp_rank", "?")
                if gp_type == "U":
                    flavs.append(gp_rank)
                    gps_dim += gp_rank**2
                else:
                    QMessageBox.warning(self, "No Subgroup Found", "Currently, Higgs global symmetry subgroup calculation is only implemented for U/SU flavour nodes.")
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
                    show_warning_with_link(self, "No Subgroup Found", r"Currently, Coulomb global symmetry subgroup calculation is not supported for balanced SU gauge groups. See Appendix D of <a href=https://doi.org/10.1007/JHEP03(2021)242>https://doi.org/10.1007/JHEP03(2021)242</a> for details on such calculations.")
                    return
                elif gp_type == "U" and gp_bal > 0:
                    gps.append(("U", 1))
                    G_removed_overb.remove_node(n)
                elif gp_type == "SU" and gp_bal > 0:
                    G_removed_overb.remove_node(n)
            elif d.get("flav_gauge") == "flavour":
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



    @staticmethod
    def _not_implemented():
        print("[TODO] button clicked – implement algorithm here")

        




def compute_balance( g: QuiverGraph, node_id: int) -> int | None:
    data = g.nodes[node_id]
    gp_type   = data.get("gp_type")
    gp_rank   = data.get("gp_rank")
    flav_flag = data.get("flav_gauge")

    if flav_flag != "gauge": #or gp_type not in ("U", "SU") or not isinstance(gp_rank, int):
        return None

    nf = 0
    for nbh in g.neighbors(node_id):
        edge_mult = g.number_of_edges(node_id, nbh)
        nbh_type  = g.nodes[nbh].get("gp_type", 0)
        nbh_rank  = g.nodes[nbh].get("gp_rank", 0)
        if nbh_type in ["U","SU","O"]:
            nf += edge_mult * nbh_rank
        elif nbh_type in ["SO", "USp"]:
            nf += edge_mult * nbh_rank/2
        

    if gp_type == "U":
        return nf - 2 * gp_rank
    elif gp_type == "SU":
        return nf - 2 * gp_rank + 1
    elif gp_type == "USp":
        return nf - gp_rank - 1
    elif gp_type == "SO":
        return nf - gp_rank + 1
