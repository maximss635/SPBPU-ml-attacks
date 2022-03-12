from main_window import *


if __name__ == "__main__":
    app = QApplication(argv)
    
    window = MainWindow()
    window.show()

    exit(app.exec())
