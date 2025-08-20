from PySide6.QtWidgets import (QVBoxLayout,
                               QPushButton, 
                               QMessageBox, QDialog, QTextEdit)
from PySide6.QtCore   import Qt

from constants import SCENE_WIDTH, SCENE_HEIGHT

def show_scrollable_message(self, title: str, message: str, font_size: int = 24):
    dialog = QDialog(self)
    dialog.setWindowTitle(title)
    dialog.resize(SCENE_WIDTH/1.25, SCENE_HEIGHT/2)  # Set desired size

    layout = QVBoxLayout(dialog)

    # Scrollable text area with larger font
    text_box = QTextEdit()
    text_box.setReadOnly(True)
    text_box.setText(message)
    text_box.setStyleSheet(f"font-size: {font_size}px;")  # Enlarge text
    layout.addWidget(text_box)

    # OK button
    ok_button = QPushButton("OK")
    ok_button.clicked.connect(dialog.accept)
    layout.addWidget(ok_button)

    dialog.exec()


def show_warning_with_link(self, title: str, message: str):
    box = QMessageBox(self)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle(title)
    box.setTextFormat(Qt.RichText)
    box.setText(message)
    box.setStandardButtons(QMessageBox.Ok)
    box.setTextInteractionFlags(Qt.TextBrowserInteraction)  # Enables clickable link
    box.exec()
