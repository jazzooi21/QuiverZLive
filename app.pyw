import sys, os
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QProxyStyle, QStyle
from window_main import MainWindow


# The following hack is needed on windows in order to show the icon in the taskbar
# See https://stackoverflow.com/questions/1551605/how-to-set-applications-taskbar-icon-in-windows-7/1552105#1552105
if os.name == 'nt':
    import ctypes
    myappid = 'qvz.0.0.1' # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)  # type: ignore

def resource_path(rel):
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel)



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

    app.setWindowIcon(QIcon(resource_path("logo.ico")))

    app.setStyle(InstantTooltipStyle())


    win = MainWindow()

    try:
        win.setWindowIcon(QIcon(resource_path("logo.ico")))
    except Exception:
        pass

    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":   # `python -m quiverzlive.app`
    main()
