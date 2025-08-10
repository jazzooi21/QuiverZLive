import sys
from PySide6.QtWidgets import QApplication, QProxyStyle, QStyle
from window_main import MainWindow

# this custom style allows tooltips to appear instantly without delay
class InstantTooltipStyle(QProxyStyle):
    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.SH_ToolTip_WakeUpDelay:
            return 0  # show instantly
        if hint == QStyle.SH_ToolTip_FallAsleepDelay:
            return 0  # hide instantly
        return super().styleHint(hint, option, widget, returnData)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle(InstantTooltipStyle())
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":   # `python -m quiverzlive.app`
    main()