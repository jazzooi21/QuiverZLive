from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QComboBox, QGraphicsView,
                               QMessageBox,
                               QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsLineItem)
from PySide6.QtGui import QPainter
from PySide6.QtCore import Qt

from quiver_scene import QuiverScene
from calculations_window import CalculationsWindow

import networkx as nx

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuiverZLive")

        # central scene & view
        self.scene = QuiverScene()
        self.view  = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)

        # ----- side panel widgets -----
        panel    = QWidget()
        p_layout = QVBoxLayout(panel); p_layout.setAlignment(Qt.AlignTop)
        
        # --- Mode buttons ---
        mode_layout = QHBoxLayout()
        self.cursor_btn = QPushButton("Cursor")
        self.addnode_btn = QPushButton("Add Node/Edge")
        self.cursor_btn.setCheckable(True)
        self.addnode_btn.setCheckable(True)
        mode_layout.addWidget(self.cursor_btn)
        mode_layout.addWidget(self.addnode_btn)
        p_layout.addLayout(mode_layout)
        
        # Set default mode
        self.cursor_btn.setChecked(True)
        self.scene.set_interaction_mode("cursor")
        self.addnode_btn.setEnabled(True)
        # Connect signals
        self.cursor_btn.clicked.connect(lambda: self.set_mode("cursor"))
        self.addnode_btn.clicked.connect(lambda: self.set_mode("add"))

        # node type buttons
        p_layout.addWidget(QLabel("Node Type:"))
        node_type_layout = QHBoxLayout()
        self.gauge_btn = QPushButton("Gauge")
        self.flavour_btn = QPushButton("Flavour")
        self.gauge_btn.setCheckable(True)
        self.flavour_btn.setCheckable(True)
        self.gauge_btn.setChecked(True)  # default to Gauge
        node_type_layout.addWidget(self.gauge_btn)
        node_type_layout.addWidget(self.flavour_btn)
        p_layout.addLayout(node_type_layout)

        # Connect signals
        self.gauge_btn.clicked.connect(lambda: self.set_node_type_button("Gauge"))
        self.flavour_btn.clicked.connect(lambda: self.set_node_type_button("Flavour"))


        # Lie group selectors
        p_layout.addWidget(QLabel("Group:"))
        group_layout = QHBoxLayout()
        self.group_type_combo = QComboBox()
        self.group_type_combo.addItems(["U", "SU", "SO", "USp"])
        group_layout.addWidget(self.group_type_combo)
        self.group_number_combo = QComboBox()
        self.group_number_combo.addItems([str(i) for i in range(1, 13)])
        group_layout.addWidget(self.group_number_combo)
        p_layout.addLayout(group_layout)
        self.group_type_combo.currentTextChanged.connect(self._enforce_even_for_usp)

        # --- calculations button ---
        self.calc_btn = QPushButton("Quiver Calculations")
        self.calc_btn.clicked.connect(self._open_calc_window)
        p_layout.addSpacing(10)
        p_layout.addWidget(self.calc_btn)

        
        # ----- menus -----
        file_menu = self.menuBar().addMenu("File")

        exit_act  = file_menu.addAction("Exit")
        exit_act.triggered.connect(self.close)



        edit_menu = self.menuBar().addMenu("Edit")

        clear_act = edit_menu.addAction("Clear All")
        clear_act.triggered.connect(self.scene.clear)

        delete_act = edit_menu.addAction("Delete Selected")
        delete_act.setShortcuts(["Del", "Backspace"])
        delete_act.triggered.connect(self._delete_selected_items)


        # ----- overall layout -----
        main = QWidget()
        main_lay = QHBoxLayout(main)
        main_lay.addWidget(self.view, 4)  # stretch factors: 4 vs 1
        main_lay.addWidget(panel,   1)
        self.setCentralWidget(main)
        self.resize(1000, 650)

    # ----------------- UI callbacks -----------------
    def set_node_type_button(self, text):
        if text == "Gauge":
            self.gauge_btn.setChecked(True)
            self.flavour_btn.setChecked(False)
            self._enable_all_groups()
        else:
            self.gauge_btn.setChecked(False)
            self.flavour_btn.setChecked(True)
            self._grey_out_su_for_flavour()
        self.scene.set_node_type(text.lower())  # "gauge" or "flavour"

    def set_mode(self, mode: str):
        if mode == "cursor":
            self.cursor_btn.setChecked(True)
            self.addnode_btn.setChecked(False)
            self.view.setDragMode(QGraphicsView.RubberBandDrag)

        else:
            self.cursor_btn.setChecked(False)
            self.addnode_btn.setChecked(True)
            self.view.setDragMode(QGraphicsView.NoDrag)

        self.scene.set_interaction_mode(mode)

    def _delete_selected_items(self):
        for item in self.scene.selectedItems():
            if isinstance(item, (QGraphicsEllipseItem, QGraphicsRectItem)):
                self.scene._remove_node(item)
            elif isinstance(item, QGraphicsLineItem):
                self.scene._remove_edge(item)

    # ------------- combobox helpers -------------

    def _enforce_even_for_usp(self):
        is_usp = self.group_type_combo.currentText() == "USp"
        for i in range(self.group_number_combo.count()):
            num = int(self.group_number_combo.itemText(i))
            self.group_number_combo.model().item(i).setEnabled(not (is_usp and num % 2 == 1))

    def _grey_out_su_for_flavour(self):
        for i in range(self.group_type_combo.count()):
            txt = self.group_type_combo.itemText(i)
            self.group_type_combo.model().item(i).setEnabled(txt != "SU")
        if self.group_type_combo.currentText() == "SU":
            # switch to first enabled
            for i in range(self.group_type_combo.count()):
                if self.group_type_combo.model().item(i).isEnabled():
                    self.group_type_combo.setCurrentIndex(i); break

    def _enable_all_groups(self):
        for i in range(self.group_type_combo.count()):
            self.group_type_combo.model().item(i).setEnabled(True)

    def _open_calc_window(self):

        if self.scene.graph.number_of_nodes() == 0:
            QMessageBox.warning(
                self, "Null Quiver",
                "Quiver is empty.")
            return
        
        elif not nx.is_connected(self.scene.graph):
            QMessageBox.warning(
                self, "Quiver Disconnected",
                "Quiver is not fully connected."
            )
            return
        
        
        flav_deg_geq2 = [
            n for n, d in self.scene.graph.nodes(data=True)
            if d.get("flav_gauge") == "flavour"
               and self.scene.graph.degree(n) > 1 ]
        if flav_deg_geq2:
            QMessageBox.warning(
                self, "Invalid Quiver",
                f"There are flavour node(s) connected to more than one node.\nIf there is a single edge of multiplicity greater than one, please amend flavour node."
            )
            return


        # If already visible, warn and bail
        if getattr(self, "_calc_win", None) and self._calc_win.isVisible():
            QMessageBox.warning(
                self, "Already Open",
                "A calculation window is already open."
            )
            self._calc_win.raise_()
            return

        # 2) Create the new window
        calc_win = CalculationsWindow(self.scene.graph, parent=self)
        calc_win.destroyed.connect(lambda obj=None: setattr(self, "_calc_win", None))
        calc_win.show()
        self._calc_win = calc_win


