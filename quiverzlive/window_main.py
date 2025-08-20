from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QComboBox, QGraphicsView,
                               QMessageBox,
                               QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsLineItem,
                               QDialog, QDialogButtonBox, QFormLayout, QLineEdit)
from PySide6.QtGui import QPainter
from PySide6.QtCore import Qt

import networkx as nx
import re
import inspect

from .quiver_scene import QuiverScene
from .window_calculations import CalculationsWindow



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
        
        # --------------------- Mode buttons ---------------------
        p_layout.addWidget(QLabel("Mouse Mode:"))
        mode_layout = QHBoxLayout()
        self.cursor_btn = QPushButton("Cursor")
        self.addnode_btn = QPushButton("Add Node/Edge")
        self.addnode_btn.setToolTip("Left click to add node.\nRight click and drag between two existing nodes to to add edge.")
        self.cursor_btn.setCheckable(True)
        self.addnode_btn.setCheckable(True)
        mode_layout.addWidget(self.cursor_btn)
        mode_layout.addWidget(self.addnode_btn)
        p_layout.addLayout(mode_layout)
        
        # Set default mode
        self.addnode_btn.setChecked(True)
        self.scene.set_interaction_mode("add")

        # Connect signals
        self.cursor_btn.clicked.connect(lambda: self.set_mode("cursor"))
        self.addnode_btn.clicked.connect(lambda: self.set_mode("add"))

        # --------------------- node type buttons ---------------------
        p_layout.addWidget(QLabel("Node Type:"))
        node_type_layout = QHBoxLayout()
        self.gauge_btn = QPushButton("Gauge")
        self.flavour_btn = QPushButton("Flavour")
        
        self.gauge_btn.setCheckable(True)
        self.flavour_btn.setCheckable(True)
        self.gauge_btn.setChecked(True)  # default: Gauge

        node_type_layout.addWidget(self.gauge_btn)
        node_type_layout.addWidget(self.flavour_btn)
        p_layout.addLayout(node_type_layout)

        self.gauge_btn.clicked.connect(lambda: self.set_node_type_button("Gauge"))
        self.flavour_btn.clicked.connect(lambda: self.set_node_type_button("Flavour"))
        
        self.gauge_btn.setShortcut("G")
        self.flavour_btn.setShortcut("F")
        
        self.gauge_btn.setToolTip("Shortcut: G")
        self.flavour_btn.setToolTip("Shortcut: F")

        # --------------------- Lie group selectors ---------------------
        p_layout.addWidget(QLabel("Group:"))
        group_layout = QHBoxLayout()
        self.group_type_combo = QComboBox()
        self.group_type_combo.addItems(["U", "SU", "SO", "USp"])
        self.group_type_combo.currentTextChanged.connect(self._enforce_even_for_usp)
        group_layout.addWidget(self.group_type_combo)
    

        self.group_number_combo = QComboBox()
        self.group_number_combo.setEditable(True)
        self.group_number_combo.lineEdit().editingFinished.connect(self._validate_group_number)
        self.group_number_combo.addItems([str(i) for i in range(1, 10+1)])
        group_layout.addWidget(self.group_number_combo)

        p_layout.addLayout(group_layout)
        


        # --------------------- Linear Quiver Input Button ------------------------
        self.linear_quiver_btn = QPushButton("Input Linear Quiver")
        self.linear_quiver_btn.setToolTip("Quickly create a linear quiver by specifying node sequence.")
        p_layout.addWidget(self.linear_quiver_btn)
        self.linear_quiver_btn.clicked.connect(self._input_linear_quiver)


        # --------------------- calculations button ---------------------
        self.calc_btn = QPushButton("Quiver Calculations")
        self.calc_btn.clicked.connect(self._open_calc_window)
        p_layout.addSpacing(10)
        p_layout.addWidget(self.calc_btn)
        self.calc_btn.setShortcut("Return")

        
        # ----- menus -----
        file_menu = self.menuBar().addMenu("File")

        exit_act  = file_menu.addAction("Exit")
        exit_act.triggered.connect(self.close)


        edit_menu = self.menuBar().addMenu("Edit")

        # undo_act = edit_menu.addAction("Undo")
        # undo_act.setShortcut("Ctrl+Z")
        # undo_act.triggered.connect(self.scene.undo)

        # redo_act = edit_menu.addAction("Redo")
        # redo_act.setShortcuts(["Ctrl+Y", "Ctrl+Shift+Z"])
        # redo_act.triggered.connect(self.scene.redo)

        select_all_act = edit_menu.addAction("Select All")
        select_all_act.setShortcut("Ctrl+A")
        select_all_act.triggered.connect(lambda: [item.setSelected(True) for item in self.scene.items()])

        delete_act = edit_menu.addAction("Delete Selected")
        delete_act.setShortcuts(["Del", "Backspace"])
        delete_act.triggered.connect(self._delete_selected_items)

        clear_act = edit_menu.addAction("Clear All")
        clear_act.triggered.connect(self.scene.clear)


        # ----- overall layout -----
        main = QWidget()
        main_lay = QHBoxLayout(main)
        main_lay.addWidget(self.view, 4)  # stretch factors: 4 vs 1
        main_lay.addWidget(panel, 1)
        self.setCentralWidget(main)
        self.resize(1000, 650)

    # ----------------- UI callbacks -----------------
    def set_node_type_button(self, text):
        if text == "Gauge":
            self.gauge_btn.setChecked(True)
            self.flavour_btn.setChecked(False)
            self._enable_all_groups()
        elif text == "Flavour":
            self.gauge_btn.setChecked(False)
            self.flavour_btn.setChecked(True)
            self._grey_out_su_for_flavour()
        lower_txt = text.lower()
        node_type = "gauge" if lower_txt=="gauge" else "flav" if lower_txt=="flavour" else None
        self.scene.set_node_type(node_type)  # "gauge" or "flavour"

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


    def _validate_group_number(self):
        text = self.group_number_combo.currentText()
        try:
            value = int(text)
            if value < 1:
                raise ValueError
        except ValueError:
            QMessageBox.warning(
            self, "Invalid Group Number",
            "Please enter a positive integer for the group number."
            )
            # Reset to previous valid value
            self.group_number_combo.setCurrentText("1")
            if self.group_type_combo.currentText() == "USp" and value % 2 != 0:
                QMessageBox.warning(
                    self, "Invalid Group Number",
                    "For USp groups, the group number must be even."
                )
                self.group_number_combo.setCurrentText("2")
            else:
                # If valid, update the combo box
                self.group_number_combo.setCurrentText(text)


    def _create_linear_quiver(self, dialog):
        self.scene.graph.clear()

        # Get the node sequence string
        node_seq_edit = dialog.findChild(QLineEdit)
        seq_text = node_seq_edit.text().strip()

        # Split into alternating nodes and bonds
        parts = re.split(r'(-|=)', seq_text)
        parts = [p for p in parts if p.strip() != '']  # remove empty

        self.scene.clear()
        prev_node = None
        prev_bond = None
        id = 0

        prev_gp_type = None  # Track previous node's group type

        for p in parts:
            if p in ("-", "="):  # bond
                prev_bond = "double" if p == "=" else "single" if p == "-" else None
                continue

            # Node processing
            if p.startswith("(") and p.endswith(")"):
                node_type = "gauge"
            elif p.startswith("[") and p.endswith("]"):
                node_type = "flav"
            else:
                continue  # Should not happen if validated

            inner = p[1:-1]
            match = re.match(r"([A-Za-z]*)(\d+)$", inner)

            gp_type = match.group(1) if match.group(1) else "U"
            gp_rank = int(match.group(2))

            node_id = id
            id += 1
            self.scene.graph.add_node(node_id, gp_type=gp_type, gp_rank=gp_rank, node_type=node_type)

            if prev_node is not None:
                # Validation: USp-USp or SO-SO not allowed
                prev_attrs = self.scene.graph.nodes[prev_node]
                if (gp_type == "USp" and prev_attrs.get("gp_type") == "USp") or \
                   (gp_type == "SO" and prev_attrs.get("gp_type") == "SO"):
                    QMessageBox.warning(
                        self, "Invalid Orthosymplectic quivers",
                        "For orthosymplectic quivers, SO and USp nodes must alternate.\n" \
                        "SO-SO and USp-USp bonds are not allowed."
                    )
                    self.scene.graph.clear()
                    return

                # Add bond type as edge attribute
                if prev_bond == "single":
                    self.scene.graph.add_edge(prev_node, node_id)
                if prev_bond == "double":
                    self.scene.graph.add_edge(prev_node, node_id)
                    self.scene.graph.add_edge(prev_node, node_id)
            prev_node = node_id
            prev_bond = None

        self._open_calc_window()


    def _input_linear_quiver(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Input Linear Quiver")
        layout = QFormLayout(dialog)
        info_label = QLabel("Format:\n"
                            "flavour group in []\n"
                            "gauge group in ()\n"
                            "'-' for single bond, '=' for double bond\n"
                            "Example: [1]-(1)=(2)-[2], [1]-(U1)=(SO2)-[2]")
        layout.addRow(info_label)

        node_seq_edit = QLineEdit()
        node_seq_edit.setPlaceholderText("e.g. (2)-[1]-(1)=(2)-[2]")
        layout.addRow("Node sequence", node_seq_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        def accept():   
            seq_text = node_seq_edit.text().strip()
            if not seq_text:
                QMessageBox.warning(dialog, "Empty Input", "Please enter a node sequence.")
                return

            # Split keeping bonds
            parts = re.split(r'(-|=)', seq_text)
            parts = [p for p in parts if p.strip() != '']

            expect_node = True
            for p in parts:
                if expect_node:
                    if not (p.startswith("(") and p.endswith(")") or p.startswith("[") and p.endswith("]")):
                        QMessageBox.warning(dialog, "Input Error", "Invalid node format.")
                        return
                    inner = p[1:-1]
                    match = re.match(r"([A-Za-z]*)(\d+)$", inner)
                    if not match:
                        QMessageBox.warning(dialog, "Input Error", "Invalid node content.")
                        return
                    gp_type = match.group(1)
                    gp_rank = int(match.group(2))
                    valid_groups = {"U", "SU", "SO", "USp", ""}
                    if gp_type not in valid_groups:
                        QMessageBox.warning(dialog, "Input Error", f"Group type not recognised: {gp_type}")
                        return
                    if gp_rank < 1:
                        QMessageBox.warning(dialog, "Input Error", "Group rank must be positive.")
                        return
                    if gp_type == "USp" and gp_rank % 2 != 0:
                        QMessageBox.warning(dialog, "Input Error", "USp groups must have even rank.")
                        return
                    expect_node = False
                else:
                    if p not in ("-", "="):
                        QMessageBox.warning(dialog, "Input Error", "Invalid bond symbol (use - or =).")
                        return
                    expect_node = True

            dialog.accept()
            self._create_linear_quiver(dialog)

        buttons.accepted.connect(accept)
        buttons.rejected.connect(dialog.reject)
        dialog.exec()





    def _open_calc_window(self):
        # for bugfixing
        # print(self.scene.graph.nodes(data=True)) 
        

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
            if d.get("node_type") == "flav"
               and self.scene.graph.degree(n) > 1 
            ]
        if flav_deg_geq2:
            QMessageBox.warning(
                self, "Invalid Quiver",
                f"There are flavour node(s) connected to more than one node.\n"\
                f"If there is a single edge of multiplicity greater than one, please amend flavour node."
            )
            return
        
        # Check for adjacent USp-USp or SO-SO nodes
        for u, v in self.scene.graph.edges():
            u_type = self.scene.graph.nodes[u].get("gp_type")
            v_type = self.scene.graph.nodes[v].get("gp_type")
            if (u_type == "USp" and v_type == "USp") or (u_type == "SO" and v_type == "SO"):
                QMessageBox.warning(
                    self, "Invalid Orthosymplectic Quiver",
                    "For orthosymplectic quivers, SO and USp nodes must alternate.\n"
                    "SO-SO and USp-USp bonds are not allowed."
                )
                return


        # If already visible, warn and abort
        if getattr(self, "_calc_win", None) and self._calc_win.isVisible():
            QMessageBox.warning(
                self, "Already Open",
                "A calculation window is already open."
            )
            self._calc_win.raise_()
            return

        # Only ask about layout if not called from _input_linear_quiver
        for n in self.scene.graph.nodes:
            if "pos" in self.scene.graph.nodes[n]:
                del self.scene.graph.nodes[n]["pos"]

        caller = inspect.stack()[1].function
        if caller != "_input_linear_quiver":
            reply = QMessageBox.question(
            self,
            "Node Positions",
            "Do you want to use automatic layout?\n\n"
            "Yes: Use automatic layout\n"
            "No: Keep original positions",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
            )

            use_original_pos = (reply == QMessageBox.No)
            
        else:
            use_original_pos = False

        if use_original_pos:
            pos = {n: (self.scene.graph.nodes[n].get("x"), self.scene.graph.nodes[n].get("y")) for n in self.scene.graph.nodes}
            nx.set_node_attributes(self.scene.graph, pos, "pos")

            

        # Create new window
        calc_win = CalculationsWindow(self.scene.graph, parent=self)
        calc_win.destroyed.connect(lambda obj=None: setattr(self, "_calc_win", None))
        calc_win.show()
        self._calc_win = calc_win


