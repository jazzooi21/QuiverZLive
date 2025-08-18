from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QCheckBox, QGraphicsView,
                               QMessageBox, QDialog, QTextEdit, QInputDialog,
                               QLabel, QScrollArea)
from PySide6.QtGui    import QPen, QBrush, QColor, QPainter
from PySide6.QtCore   import QRectF, Qt, QLineF



from constants import NODE_RADIUS, PEN_NODE, SCENE_WIDTH, SCENE_HEIGHT

from Qt_zoomable_view import ZoomableGraphicsView
from window_hasse import HasseDiagramView
from static_scene import _StaticScene

from graph_model import QuiverGraph, compute_balance, mixU_linear, QG_to_Mv, Mv_to_QG
from nx_dynkin import Dynkin_A, Dynkin_D, Dynkin_E
from nx_layouts import plot_caterpillar, plot_sunshine, plot_sunshine_multicycles
from Qt_custom_boxes import show_scrollable_message, show_warning_with_link


from calc_HS_C import hilbert_series_from_quiver_graph
from calc_linearmirror import MagneticQuiver
from calc_hasse import fission_decay


import networkx as nx
import copy
from collections import Counter
import time
import numpy as np



class CalculationsWindow(QMainWindow):
    def __init__(self, g: QuiverGraph, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Quiver Calculations")
        self.graph = g                # reference only; NOT copied
        self.scene = _StaticScene(self.graph)
        self.view  = ZoomableGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.resize(1000, 650)

        
        # --- pos if no pos ----
        if not all("pos" in d for _, d in g.nodes(data=True)):
            cycles = nx.cycle_basis(nx.Graph(g))
            if not cycles:
                pos = plot_caterpillar(
                    g,
                    hsep=5 * NODE_RADIUS + 10,
                    vsep=5 * NODE_RADIUS + 10,
                )
            elif len(cycles) == 1:
                pos = plot_sunshine(
                    g,
                    radius=5 * NODE_RADIUS + 10,
                    vsep=5 * NODE_RADIUS + 10,
                )
            else:
                pos = plot_sunshine_multicycles(g)
                scale = 5 * NODE_RADIUS + 10
                for k in pos:
                    pos[k] = (pos[k][0] * scale, pos[k][1] * scale)

            for n, (x, y) in pos.items():
                self.graph.nodes[n]["pos"] = (x, y)

        # Print node positions for debugging
        # for n, d in self.graph.nodes(data=True):
        #     print(f"Node {n}: pos = {d.get('pos')}")




        # --- side controls ---
        
        side = QWidget()
        s_lay = QVBoxLayout(side)
        s_lay.setAlignment(Qt.AlignTop)        

        # -------- Adjacency matrix button ---------------------
        self.show_adj_btn = QPushButton("Show Adjacency Matrix")
        self.show_adj_btn.setToolTip("Display the adjacency matrix and node ranks vector of the current quiver.")
        self.show_adj_btn.clicked.connect(self._calc_adjM)

        if any(d.get("node_type") == "flav" for _, d in self.graph.nodes(data=True)):
            self.show_adj_btn.setEnabled(False)
            self.show_adj_btn.setToolTip("Calculate adjacency matrix and ranks vector for the quiver.\n" \
                                              "Only valid for unframed (no flavours) unitary quivers.")
            
        elif any(d.get("node_type") == "gauge" and d.get("gp_type") != "U" for _, d in self.graph.nodes(data=True)):
            self.show_adj_btn.setEnabled(False)
            self.show_adj_btn.setToolTip("Calculate adjacency matrix and ranks vector for the quiver.\n" \
                                              "Only valid for unframed (no flavours) unitary quivers.")


        # --------------------- Gauge/Ungauge buttons ---------------------
        self.gauge_btn = QPushButton("Gauge")
        self.gauge_btn.clicked.connect(self._gauge)
        self.ungauge_btn = QPushButton("Ungauge")
        self.ungauge_btn.clicked.connect(self._ungauge)
        flavour_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "flav"]
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
            
        # --------------------- Combine flavours button ---------------------
        self.combine_btn = QPushButton("Combine Flavours")
        self.combine_btn.setToolTip("If multiple unitary flavour nodes are connected to the same gauge node,\n"
                                              "combine them into one flavour node whose rank is sum of originals.")
        self.combine_btn.clicked.connect(self._combine_flavours)


        # --------------------- Node colour mode buttons ---------------------
        self.colour_none_btn = QPushButton("None")
        self.colour_balance_btn = QPushButton("Balance")
        self.colour_group_btn = QPushButton("Group")

        self.colour_none_btn.setCheckable(True)
        self.colour_balance_btn.setCheckable(True)
        self.colour_group_btn.setCheckable(True)

        self.colour_none_btn.setChecked(True) # default none

        self.colour_none_btn.setToolTip("All nodes white.")
        self.colour_balance_btn.setToolTip("red = underbalanced\n" \
                                          "white = balanced\n" \
                                          "gray = overbalanced")
        self.colour_group_btn.setToolTip("white = U (unitary)\n" \
                                        "gray = SU (special unitary)\n" \
                                        "red = SO (special orthogonal)\n" \
                                        "blue = USp (symplectic)")
        # Default: Balance colouring
        self.colour_none_btn.clicked.connect(lambda: self._colours("none"))
        self.colour_balance_btn.clicked.connect(lambda: self._colours("bal"))
        self.colour_group_btn.clicked.connect(lambda: self._colours("gp"))

        

        # --------------------- Global symm subgp buttons ---------------------
        self.glob_symm_subgp_C_btn = QPushButton("Coulomb")
        self.glob_symm_subgp_C_btn.setToolTip("Calculate Coulomb Branch global symmetry subgroup.\n"
                                              "(this may or may not be the global symmetry group)\n"
                                              "Currently only developed for U and SU gauge nodes.")
        self.glob_symm_subgp_C_btn.clicked.connect(self._glob_symm_subgp_C)

        self.glob_symm_subgp_H_btn = QPushButton("Higgs")
        self.glob_symm_subgp_H_btn.setToolTip("Calculate Higgs Branch global symmetry subgroup.\n"
                                              "(this may or may not be the global symmetry group)")
        self.glob_symm_subgp_H_btn.clicked.connect(self._glob_symm_subgp_H)

        gauge_nodes = [d for n, d in self.graph.nodes(data=True) if d.get("node_type") == "gauge"]    
        flav_nodes = [d for n, d in self.graph.nodes(data=True) if d.get("node_type") == "flav"]
        nodes = [d for n, d in self.graph.nodes(data=True)]

        if any(d.get("gp_type") in ("SO","USp") for d in nodes) and any(d.get("gp_type") in ("U","SU") for d in nodes):
            self.glob_symm_subgp_H_btn.setEnabled(False)
            self.glob_symm_subgp_H_btn.setToolTip("Calculate Higgs Branch global symmetry subgroup.\n"
                                              "(this may or may not be the global symmetry group)\n"
                                              "Currently only developed for unitary or orthosymplectic gauge nodes.\n"
                                              "Not implemented for unitary-orthosymplectic quivers.")
            

        # --------------------- HS buttons ---------------------
        self.HS_C_btn = QPushButton("Coulomb")
        self.HS_C_btn.setToolTip("Calculate the Hilbert Series of the Coulomb branch for the current quiver.")
        self.HS_C_btn.clicked.connect(self._HS_C)

        self.HS_H_btn = QPushButton("Higgs")
        self.HS_H_btn.setToolTip("Calculate the Hilbert Series of the Higgs branch for the current quiver.")
        self.HS_H_btn.setEnabled(False)

        # --------------------- 3d Mirror Button ---------------------
        self.linear_mirror_btn = QPushButton("Find 3d Mirror")
        self.linear_mirror_btn.setToolTip("Attempt to find the 3d mirror of the current quiver.\n"
                                   "Currently implemented for linear mixed unitary quivers.")
        self.linear_mirror_btn.clicked.connect(self._linear_3d_mirror)

        if mixU_linear(self.graph): # quiver is linear and mixed unitary
            self.linear_mirror_btn.setEnabled(True)
            self.linear_mirror_btn.setToolTip("Find the (ungauged) 3d mirror of this linear mixed unitary quiver.")
        else: 
            self.linear_mirror_btn.setEnabled(False)
            self.linear_mirror_btn.setToolTip("3d mirror calculation is only implemented for linear, mixed unitary quivers.")

        
        # ------------------ Hasse Button ---------------------
        self.hasse_btn = QPushButton("Hasse Diagram")
        self.hasse_btn.setToolTip("Calculate Hasse diagram.")
        self.hasse_btn.clicked.connect(self._hasse)
        # Only valid for unframed unitary quivers
        if any(d.get("node_type") == "flav" for _, d in self.graph.nodes(data=True)):
            self.hasse_btn.setEnabled(False)
            self.hasse_btn.setToolTip("Calculate adjacency matrix and ranks vector for the quiver.\n" \
                                              "Only valid for unframed (no flavours) unitary quivers.")
        elif any(d.get("node_type") == "gauge" and d.get("gp_type") != "U" for _, d in self.graph.nodes(data=True)):
            self.hasse_btn.setEnabled(False)
            self.hasse_btn.setToolTip("Calculate adjacency matrix and ranks vector for the quiver.\n" \
                                              "Only valid for unframed (no flavours) unitary quivers.")
    


        # --------------- layout -----------------------------
        s_lay.addWidget(QLabel("Node colouring:"))
        colour_mode_row = QWidget()
        colour_mode_layout = QHBoxLayout(colour_mode_row)
        colour_mode_layout.setContentsMargins(0, 0, 0, 0)
        colour_mode_layout.addWidget(self.colour_none_btn)
        colour_mode_layout.addWidget(self.colour_balance_btn)
        colour_mode_layout.addWidget(self.colour_group_btn)
        s_lay.addWidget(colour_mode_row)

        
        s_lay.addWidget(QLabel(""))


        s_lay.addWidget(QLabel("<b>Calculations</b>"))
        
        s_lay.addWidget(self.show_adj_btn)
        self.show_adj_btn.setToolTip("Calculate adjacency matrix and ranks vector for unframed unitary quivers.")



        s_lay.addWidget(QLabel("Global Symmetry Subgroup:"))
        symm_row = QWidget()
        symm_layout = QHBoxLayout(symm_row)
        symm_layout.setContentsMargins(0, 0, 0, 0)
        symm_layout.addWidget(self.glob_symm_subgp_C_btn)
        symm_layout.addWidget(self.glob_symm_subgp_H_btn)
        s_lay.addWidget(symm_row)


        s_lay.addWidget(QLabel("Hilbert Series:"))
        hs_row = QWidget()
        hs_layout = QHBoxLayout(hs_row)
        hs_layout.setContentsMargins(0, 0, 0, 0)
        hs_layout.addWidget(self.HS_C_btn)
        hs_layout.addWidget(self.HS_H_btn)
        s_lay.addWidget(hs_row)
        

        s_lay.addWidget(QLabel(""))
        s_lay.addWidget(QLabel("<b>New Quiver</b>"))
        

        s_lay.addWidget(self.combine_btn)

        gauge_ungauge_row = QWidget()
        gauge_ungauge_layout = QHBoxLayout(gauge_ungauge_row)
        gauge_ungauge_layout.setContentsMargins(0, 0, 0, 0)
        gauge_ungauge_layout.addWidget(self.gauge_btn)
        gauge_ungauge_layout.addWidget(self.ungauge_btn)
        s_lay.addWidget(gauge_ungauge_row)


        s_lay.addWidget(self.linear_mirror_btn)

        s_lay.addWidget(self.hasse_btn)

        # --- main layout ---
        central = QWidget(); lay = QHBoxLayout(central)
        lay.addWidget(self.view, 4); lay.addWidget(side, 1)
        self.setCentralWidget(central); self.resize(800, 500)


    def _calc_adjM(self):
        """
        Display the adjacency matrix and node ranks vector of the current quiver.
        Uses QG_to_Mv for matrix and vector extraction.
        """

        g = self.graph
        M, ranks = QG_to_Mv(g)

        # Format the adjacency matrix as a readable HTML table
        matrix_html = "<table border='1' cellspacing='0' cellpadding='2' style='font-family:monospace;font-size:10pt;'>"
        for row in M.tolist():
            matrix_html += "<tr>" + "".join(f"<td align='center'>{val}</td>" for val in row) + "</tr>"
        matrix_html += "</table>"

        # Format the node ranks vector
        ranks_html = ", ".join(str(r) for r in ranks)

        # Combine into HTML
        html = (
            f"<b>Adjacency Matrix:</b>{matrix_html}<br><br>"
            f"<b>Node Ranks:</b><br>[{ranks_html}]"
        )

        show_scrollable_message(
            self,
            "Adjacency Matrix and Node Ranks",
            html
        )


    def _combine_flavours(self):
        """
        Create a **new** CalculationsWindow in which any gauge node that had
        multiple connected U-flavour nodes is now connected to exactly one
        U-flavour node whose rank is the sum of the originals.
        """
        new_g = copy.deepcopy(self.graph)           # work on a fresh copy

        # Generate a fresh node id counter
        next_id = max(new_g.nodes, default=-1) + 1

        for gauge in list(new_g.nodes):          # snapshot of original IDs
            if gauge not in new_g:               
                continue      
            ndata = new_g.nodes[gauge]
            if ndata.get("node_type") != "gauge":
                continue  # skip non-gauge nodes

            # All directly-connected U-flavour neighbours
            u_flavours = [
                n for n in new_g.neighbors(gauge)
                if new_g.nodes[n].get("node_type") == "flav"
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
                node_type="flav",
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


    def _colours(self, state: str):

        if state == "none":
            self.colour_none_btn.setChecked(True)
            self.colour_balance_btn.setChecked(False)
            self.colour_group_btn.setChecked(False)
        elif state == "bal":
            self.colour_none_btn.setChecked(False)
            self.colour_balance_btn.setChecked(True)
            self.colour_group_btn.setChecked(False)
        elif state == "gp":
            self.colour_none_btn.setChecked(False)
            self.colour_balance_btn.setChecked(False)
            self.colour_group_btn.setChecked(True)
        
        self.scene.set_colour_mode(state)

        """Re-colour nodes according to balance when box is ticked."""
        for item in self.scene.items():
            nid = item.data(0)
            if nid is None:                       # skip edges / labels
                continue
            
            if state == 'none':                # colouring OFF
                colour = Qt.white

            elif state == 'bal':
                bal = compute_balance(self.graph, nid)
                if bal is None: #flav
                    colour = Qt.white
                elif bal < 0:
                    colour = QColor("#ff9898")    # light red
                elif bal > 0:
                    colour = QColor("#bebdbd")    # light grey
                else:
                    colour = Qt.white
            
            elif state == 'gp':
                data = self.graph.nodes[nid]
                gp_type = data.get("gp_type")
                if gp_type == "U":
                    colour = Qt.white
                elif gp_type == "SU":
                    colour = QColor("#bebdbd")  # light gray
                elif gp_type == "SO":
                    colour = QColor("#fd6b6b")  # light red
                elif gp_type == "USp":
                    colour = QColor("#5a5aef")  # light blue

            item.setBrush(QBrush(colour))


    def _glob_symm_subgp_H(self):
        
        
        gps_dim = 0



        # SU from higher multiplicity edges
        edge_mults = []
        for u, v in self.graph.edges():
            edge_mult = self.graph.number_of_edges(u, v)
            if edge_mult > 1:
                # only consider edges with multiplicity > 1, since SU(1) is trivial
                edge_mults.append(edge_mult)
        
        e_SUs = []
        for mult in edge_mults:
            e_SUs.append(("SU",mult))
            gps_dim += mult**2 - 1

        glob_symm_subgp_str_e = " × ".join(
            f"SU({n})^{m}" if m > 1 else f"SU({n})"
            for n, m in sorted(Counter(edge_mults).items(), reverse=True)) + " × "



        gauge_nodes = [d for n, d in self.graph.nodes(data=True) if d.get("node_type") == "gauge"]
        nodes = [d for n, d in self.graph.nodes(data=True)]

        # all U gauge and flav
        if all(d.get("gp_type") == "U" for d in nodes):

            gps_dim -= 1 # gauge U(1)

            flavs = []
            for n, d in self.graph.nodes(data=True):
                if d.get("node_type") == "flav":
                    gp_type = d.get("gp_type", "?")
                    gp_rank = d.get("gp_rank", "?")
                    if gp_type == "U":
                        flavs.append((gp_type,gp_rank))
                        gps_dim += gp_rank**2
                    else:
                        QMessageBox.warning(self, "No Subgroup Found", "Higgs global symmetry subgroup calculation is not implemented for this quiver.")
                        return
            
            flav_ranks = sorted([n[1] for n in flavs], reverse=True)

            if not flavs:
                QMessageBox.warning(self, "No Subgroup Found", "No flavour nodes found.")
                return
            else:
                counts = Counter(flav_ranks)
                glob_symm_subgp_str_f = " × ".join(
                    f"U({n})^{m}" if m > 1 else f"U({n})"
                    for n, m in sorted(counts.items(), reverse=True))
                
                if 1 in flav_ranks:
                    glob_symm_subgp_str_f_rem_U1 = " × ".join(f"U({n})^{m}" if m > 1 else f"U({n})" for n, m in sorted(Counter(flav_ranks[:-1]).items(), reverse=True))
                    if not glob_symm_subgp_str_f_rem_U1:
                        if edge_mults == []:
                            glob_symm_subgp_str_f_rem_U1 = r"trivial group"
                        else:
                            glob_symm_subgp_str_f_rem_U1 = glob_symm_subgp_str_f_rem_U1[:-3]  # remove trailing " × " 
                    output_str = (
                        f"{glob_symm_subgp_str_e}S(  {glob_symm_subgp_str_f}  )\n"
                        f"≅ {glob_symm_subgp_str_e}{glob_symm_subgp_str_f} / U(1)\n"
                        f"≅ {glob_symm_subgp_str_e}{glob_symm_subgp_str_f_rem_U1}\n\n"
                        f"Dimension: {gps_dim}"
                    )

                else:
                    output_str = (
                        f"{glob_symm_subgp_str_e}S(  {glob_symm_subgp_str_f}  )\n"
                        f"≅ {glob_symm_subgp_str_e}{glob_symm_subgp_str_f} / U(1)\n\n"
                        f"Dimension: {gps_dim}"
                    )

        # orthosymplectic
        elif all(d.get("gp_type") in ("SO","USp") for d in nodes):
            flavs = e_SUs
            for n, d in self.graph.nodes(data=True):
                if d.get("node_type") == "flav":
                    gp_type = d.get("gp_type", "?")
                    gp_rank = d.get("gp_rank", "?")
                    if gp_type in ("SO","USp"):
                        flavs.append((gp_type,gp_rank))
                    elif gp_type in ("U","SU"):
                        QMessageBox.warning(self, "No Subgroup Found", "Higgs global symmetry subgroup calculation is not implemented for unitary-orthosymplectic quivers.")
                        return
                    
            priority = {"SU": 0, "SO": 1, "USp": 2}
            flavs.sort(key=lambda x: (priority.get(x[0], 99), -x[1]))
            
            gps_str = []
            gps_dim = 0
            for gp in flavs:
                gp_type = gp[0]
                gp_rank = gp[1]
                if gp_type == "SO":
                    gps_str.append(f"SO({gp_rank})")
                    gps_dim += gp_rank * (gp_rank - 1) / 2
                elif gp_type == "USp":
                    gps_str.append(f"USp({gp_rank})")
                    gps_dim += gp_rank * (gp_rank + 1) / 2
            
            counts = Counter(gps_str)
            seen = set()
            flavs_str_combine = " × ".join(
                f"{g}^{counts[g]}" if counts[g] > 1 else g
                for g in gps_str if not (g in seen or seen.add(g))
            )
            
            output_str = (
                        f"{flavs_str_combine}\n\n"
                        f"Dimension: {gps_dim}"
                    )
            
        elif any(d.get("gp_type") in ("SO","USp") for d in nodes) and any(d.get("gp_type") in ("U","SU") for d in nodes):
            QMessageBox.warning(self, "No Subgroup Found", "Higgs global symmetry subgroup calculation is not implemented for unitary-orthosymplectic quivers.")
            return


        show_scrollable_message(self, "Higgs Global Symmetry Subgroup", output_str)
        return


    def _glob_symm_subgp_C(self):

        gps = []

        G_removed_overb = copy.deepcopy(self.graph)
    
        for n, d in self.graph.nodes(data=True):
            if d.get("node_type") == "gauge":
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
            elif d.get("node_type") == "flav":
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
        flavour_nodes = [n for n, d in new_g.nodes(data=True) if d.get("node_type") == "flav"]
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
        new_g.add_node(new_U1, gp_type="U", gp_rank=1, node_type="gauge")

        # ----- connect all previous flavour nodes to the new U(1) gauge node -----
        # Edge multiplicity is rank of corresponding previous flavour node
        for flav in flavour_nodes:
            neighbour = next(new_g.neighbors(flav)) # Each flavour node has only one neighbour (the gauge node it's attached to)
            flav_rank = new_g.nodes[flav].get("gp_rank")
            for _ in range(flav_rank):
                new_g.add_edge(new_U1, neighbour)

        # ----- remove node_type flag from previous flavour nodes -----
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
        flavour_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "flav"]
        if bool(flavour_nodes):  # unframed/flavourless quiver
            QMessageBox.warning(self, "Unframed Quiver", "Ungauging is only valid for unframed/flavourless quivers.")
            return

        # Check if there is a U(1) gauge node
        u1_gauge_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "gauge" and d.get("gp_type") == "U" and d.get("gp_rank") == 1]
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
                        node_type="flav"
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
            "(NOTE: Computation time scales exponentially with calculation order, number of nodes and group rank.)",
            value=5, minValue=1, maxValue=21
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
        HS_str = HS_str.replace("*", "")

        output_str = HS_str
        show_scrollable_message(self,
                                f"Coulomb branch Hilbert Series (Calculation time: {elapsed:.2f} secs)",
                                output_str
                                )
        return


    def _linear_3d_mirror(self):
        # Extract gauge nodes in order (linear quiver: path)
        # Each gauge node is U or SU, each may have at most one U-flavour attached
        # Return: [flavour_ranks], [gauge_ranks], [gauge_types]

        # 1. Find the ordered list of gauge nodes along the path
        # Only nodes of type "gauge"
        gauge_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "gauge"]

        if len(gauge_nodes) > 1:
            gauge_subgraph = self.graph.subgraph(gauge_nodes)
            endpoints = [n for n in gauge_subgraph.nodes if gauge_subgraph.degree(n) == 1]
            path = []
            visited = set()
            current = endpoints[0]
            prev = None
            while True:
                path.append(current)
                visited.add(current)
                neighbors = [n for n in gauge_subgraph.neighbors(current) if n != prev]
                if not neighbors:
                    break
                prev, current = current, neighbors[0]
        elif len(gauge_nodes) == 1:
            path = [gauge_nodes[0]]

        gauge_ranks = []
        gauge_types = []
        flav_ranks = []
        for gauge_node in path:

            d = self.graph.nodes[gauge_node]
            gauge_ranks.append(d.get("gp_rank"))
            gauge_types.append("u" if d.get("gp_type") == "U" else "s" if d.get("gp_type") == "SU" else "unknown")
            
            # Find attached U-flavour (node_type=="flav", gp_type=="U")
            flav = None
            for n in self.graph.neighbors(gauge_node):
                
                nd = self.graph.nodes[n]
                if nd.get("node_type") == "flav" and nd.get("gp_type") == "U":
                    
                    if flav is not None:
                        raise RuntimeError(f"Gauge node {gauge_node} has more than one attached U-flavour node.")

                    flav = nd.get("gp_rank")
                    
            flav_ranks.append(flav if flav is not None else 0)

        
        mq = MagneticQuiver(flav_ranks, gauge_ranks, gauge_types)
        
        class MagneticQuiverResultsDialog(QDialog):
            def __init__(self, results, parent=None):
                super().__init__(parent)
                self.setWindowTitle(f"{len(results)} Magnetic Quiver{'s' if len(results) > 1 else ''} Found")
                self.selected_quiver = None 

                layout = QVBoxLayout(self)
                scroll = QScrollArea()
                scroll.setWidgetResizable(True)
                container = QWidget()
                vbox = QVBoxLayout(container)

                for idx, (bl, Q, M, ranks) in enumerate(results):
                    group = QWidget()
                    group_layout = QHBoxLayout(group)

                    open_btn = QPushButton("Open")
                    open_btn.clicked.connect(lambda _, q=Q: self.open_quiver(q))
                    group_layout.addWidget(open_btn)

                    quiver_view = QGraphicsView(_StaticScene(Q, parent=self))
                    quiver_view.setRenderHint(QPainter.Antialiasing)
                    quiver_view.setMinimumWidth(220)
                    quiver_view.setMinimumHeight(180)
                    group_layout.addWidget(quiver_view)

                    # Format the adjacency matrix as a readable HTML table
                    matrix_html = "<table border='1' cellspacing='0' cellpadding='2' style='font-family:monospace;font-size:10pt;'>"
                    for row in M.tolist():
                        matrix_html += "<tr>" + "".join(f"<td align='center'>{val}</td>" for val in row) + "</tr>"
                    matrix_html += "</table>"
                    
                    info = QLabel(
                        f"<b>Brane locking:<br></b> {bl}<br><br>"
                        f"<b>Node Ranks:<br></b> {ranks}<br><br>"
                        f"<b>Adjacency Matrix:</b><br>{matrix_html}"
                    )
                    info.setTextFormat(Qt.RichText)

                    group_layout.addWidget(info)
                    vbox.addWidget(group)
                container.setLayout(vbox)
                scroll.setWidget(container)
                layout.addWidget(scroll)


            def open_quiver(self, quiver_graph):
                main_win = self.parent().parent()  # MainWindow
                new_win = CalculationsWindow(quiver_graph, parent=main_win)
                main_win._calc_win = new_win
                new_win.show()
                self.accept()
                self.parent().close()


        results = list(mq.magnetic_quivers_full())
        if not results:
            QMessageBox.information(self, "No Magnetic Quivers", "No valid magnetic quivers found.")
            return

        dlg = MagneticQuiverResultsDialog(results, parent=self)
        dlg.exec()


    def _hasse(self):
        class HasseDiagramDialog(QDialog):
            def __init__(self, full_qg, M, all_leaves, hasse_mult, idx2node, parent=None):
                super().__init__(parent)
                self.setWindowTitle(f"Hasse Diagram: {len(all_leaves)} Leaves [double click quiver to open new Calculations Window]")
                layout = QVBoxLayout(self)
                view = HasseDiagramView(full_qg, M, all_leaves, hasse_mult, idx2node, parent=self, open_quiver_callback=self._open_quiver)
                layout.addWidget(view)
                
            def _open_quiver(self, quiver_graph):
                main_win = self.parent().parent()  # MainWindow
                new_win = CalculationsWindow(quiver_graph, parent=main_win)
                main_win._calc_win = new_win
                new_win.show()
                self.accept()
                self.parent().close()

        # NEW: fission_decay returns 5 values
        M, ranks, idx2node, all_leaves, hasse_mult = fission_decay(self.graph)

        dlg = HasseDiagramDialog(self.graph, M, all_leaves, hasse_mult, idx2node, parent=self)

        dlg.exec()



  
    

    @staticmethod
    def _not_implemented():
        print("[TODO] button clicked – implement algorithm here")

    

