from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader
from threading import Thread
import os, sys

# ---- Main Application Object ---- #
class ZestyMenu(QMainWindow):
    def __init__(self):
        # Initialize Window
        super(ZestyMenu, self).__init__()
        self.resize(400, 400)
        self._load_ui()
        self._bind_buttons()

    def _load_ui(self):
        loader  = QUiLoader()
        ui_file = QFile("ui/main.ui")
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file, self)
        ui_file.close()
        self.setCentralWidget(self.ui.centralWidget())

    def _bind_buttons(self):
        # Bind Buttons
        self.ui.ConcatenateButton.clicked.connect(self._launch_concatenate)
        self.ui.FitWASSRButton.clicked.connect(self._launch_wassr)
        self.ui.FitCESTButton.clicked.connect(self._launch_cest)
        self.ui.InspectButton.clicked.connect(self._launch_inspect)

    def _launch_concatenate(self):
        t = Thread(target = lambda: os.system('python concat.py'))
        t.start()

    def _launch_wassr(self):
        t = Thread(target = lambda: os.system('python fitwassr.py'))
        t.start()

    def _launch_cest(self):
        t = Thread(target = lambda: os.system('python fitcest.py'))
        t.start()
    
    def _launch_inspect(self):
        t = Thread(target = lambda: os.system('python inspector.py'))
        t.start()

# ---- Launch Application ---- #
if __name__ == "__main__":
    app = QApplication(sys.argv)
    QApplication.setStyle('Fusion')
    window = ZestyMenu()
    window.show()
    sys.exit(app.exec())