from PySide6.QtWidgets import QApplication
import sys
from src.ui_layout import Ui_window
from src.detect_moudle import Detector

class MainApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = Ui_window()
        self.window.setupUi(self.window)
        self.detector = Detector(model_path="models/111.onnx")

    def run(self):
        self.window.show()
        sys.exit(self.app.exec())

if __name__ == "__main__":
    main_app = MainApp()
    main_app.run()