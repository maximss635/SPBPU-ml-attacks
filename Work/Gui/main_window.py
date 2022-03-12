from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QThread, QObject

from ui_main_window import *

from predict import *

from sys import argv


class Predicter(QThread):
    def __init__(self, window):
        super().__init__()
        self.__window = window

    def run(self):
        self.__window.predictions = predict(MainWindow.PATH_IMAGE_SAVE)


class Scene(QGraphicsScene):
    def __init__(self, window, w, h, parent=None):
        super().__init__(parent)

        self.__w = w
        self.__h = h

        self.__window = window

    
    def mousePressEvent(self, event):
        self.__window.clear_progress_bars()
        self.__previous_point = event.scenePos()
    

    def mouseMoveEvent(self, event):
        x = event.scenePos().x()
        y = event.scenePos().y()

        if (abs(x) < self.__w / 2) and \
            (abs(y) < self.__h / 2):
            
            self.addLine(
                self.__previous_point.x(),
                self.__previous_point.y(),
                x, y,
                QPen(Qt.black, MainWindow.SCENE_SCALE, 
                    Qt.SolidLine, Qt.RoundCap)
            )

        self.__previous_point = event.scenePos()


class MainWindow(QMainWindow):

    SCENE_SCALE = 8
    PATH_IMAGE_SAVE = "image.png"

    __SCENE_PIXEL_WIDTH = 28
    __SCENE_PIXEL_HEIGHT = 28
    
    def __init__(self, parent=None):
        super().__init__(parent)

        self.__ui = Ui_Widget()
        self.__ui.setupUi(self)

        self.__thread_predicter = Predicter(self)
        self.__thread_predicter.finished.connect(self.__on_new_predictions)

        __central_widget = QWidget(self)
        self.setCentralWidget(__central_widget)
        __central_widget.setLayout(self.__ui.main_layout)

        self.setFixedSize(600, 320)

        self.__progress_bar_list = [        
            self.__ui.progressBar_0,
            self.__ui.progressBar_1,
            self.__ui.progressBar_2,
            self.__ui.progressBar_3,
            self.__ui.progressBar_4,
            self.__ui.progressBar_5,
            self.__ui.progressBar_6,
            self.__ui.progressBar_7,
            self.__ui.progressBar_8,
            self.__ui.progressBar_9
        ]

        self.clear_progress_bars()

        self.__ui.graphics_view.setFixedSize(
            self.__SCENE_PIXEL_WIDTH * self.SCENE_SCALE, 
            self.__SCENE_PIXEL_HEIGHT * self.SCENE_SCALE
        )

        self.__ui.graphics_view.setLineWidth(self.SCENE_SCALE);

        self.__scene = Scene(
            self, 
            self.__ui.graphics_view.width(),
            self.__ui.graphics_view.height()
        )

        self.__ui.graphics_view.setScene(self.__scene)

        self.__ui.button_clear.clicked.connect(self.__on_button_clear)
        self.__ui.button_predict.clicked.connect(self.__on_button_predict)


    def clear_progress_bars(self):
        for pb in self.__progress_bar_list:
            pb.setValue(0)


    def __on_button_clear(self):
        self.__scene.clear()
        self.clear_progress_bars()


    def __on_button_predict(self):
        self.__save_image()
        self.__thread_predicter.start()


    def __on_new_predictions(self):
        for i, y in enumerate(self.predictions):
            self.__progress_bar_list[i].setValue(y * 100)


    def __save_image(self): 
        image = QImage( 28, 28, QImage.Format_RGB32)
        image.fill(Qt.white)

        painter = QPainter(image)

        self.__scene.render(painter)
        image.save(self.PATH_IMAGE_SAVE)

        painter.end()
