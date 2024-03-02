from scipy.fft import fft
import scipy.signal as sig
import numpy as np
import pandas as pd
import pyqtgraph as pg
import os
import time
import pyqtgraph as pg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog,QDialog, QGraphicsScene ,QLabel , QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtWidgets, uic 
from cmath import*
from numpy import *
import sys
import matplotlib
from PyQt5.QtGui import QPixmap, QImage
matplotlib.use('Qt5Agg')
from Image import Image as ig

class MyDialog(QtWidgets.QDialog):
    def __init__(self):
        super(MyDialog, self).__init__()
        # Load the UI Page
        uic.loadUi(r'mixer.ui', self)

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # Load the UI Page
        uic.loadUi(r'task4.ui', self)
        self.delta_y = 0.0
        self.delta_x = 0.0
        self.setMouseTracking(True)
        self.mousePressPosition = None
        self.mouseMovePosition = None
        #button connection
        self.component_btn.clicked.connect(self.open_dialog)
        image_graphs = [self.image1, self.image2, self.image3, self.image4]
        ft_image_graphs = [self.ft_compo_1, self.ft_compo_2, self.ft_compo_3, self.ft_compo_4]
        self.combos = [self.ft_combo1, self.ft_combo2, self.ft_combo3, self.ft_combo4]
        # Create a list to store Image instances and associated QLabel objects
        self.images = [ig(graph, ft_image, self.combos) for graph, ft_image in zip(image_graphs, ft_image_graphs)]
        # self.img=ig(self)
        #Connections
        # Connect combobox signals to the corresponding check_combo method
        self.ft_combo1.activated.connect(lambda: self.combo_activated(0))
        self.ft_combo2.activated.connect(lambda: self.combo_activated(1))
        self.ft_combo3.activated.connect(lambda: self.combo_activated(2))
        self.ft_combo4.activated.connect(lambda: self.combo_activated(3))
        # Connect double-click events to each QLabel using a loop
        for label, image_instance in zip(image_graphs, self.images):
            label.mouseDoubleClickEvent = lambda event, instance=image_instance: self.double_click_event(event, instance)

        # for label, image_instance in zip(image_graphs, self.images):
        #     label.mousePressEvent = lambda event, label_widget=label: self.mousePressEvent(event,label_widget)
        #     label.mouseMoveEvent = lambda event, label_widget=label: self.mouseMoveEvent(event,label_widget)
        #     label.mouseReleaseEvent = lambda event, instance=image_instance: self.mouseReleaseEvent(event,instance)

        for ft_image, image_instance in zip(ft_image_graphs, self.images):
            ft_image.mousePressEvent = lambda event, instance=image_instance: instance.mousePressEvent(event)
            ft_image.mouseMoveEvent = lambda event, instance=image_instance: instance.mouseMoveEvent(event)
            ft_image.mouseReleaseEvent = lambda event, instance=image_instance: instance.mouseReleaseEvent(event)
            # combo.activated.connect(lambda: self.combo_activated(ft_image, combo))


    def double_click_event(self, event, image_instance):
        if event.button() == Qt.LeftButton:
            image_instance.Browse()

    # just to calculate the initial press position
    # def mousePressEvent(self, event,label):
    #     if event.button() == Qt.LeftButton:
    #         label.mousePressPosition = event.pos()
    #         print(label.mousePressPosition)


    # def mouseMoveEvent(self, event,label):
    #     if event.buttons() == Qt.LeftButton:
    #         delta_x = event.pos().x() - label.mousePressPosition.x()
    #         delta_y = event.pos().y() - label.mousePressPosition.y()
    #         self.delta_y = delta_y
    #         self.delta_x =delta_x


    # def mouseReleaseEvent(self, event, image_instance):
    #     if event.button() == Qt.LeftButton:
    #         result = None
    #         image_instance.contrast_coef  = (-self.delta_y + 100) * 0.01
    #         result = image_instance.calculate_brightness_contrast(image_instance.image)
    #         print(result)
    #         image_instance.update_display(result)

    # def combo_activated(self, index):
    #     for i, image_instance in enumerate(self.images):
    #         if i == index:
    #             # Update the selected combo box
    #             image_instance.check_combo(index)
    #         else:
    #             # Reset other combo boxes or perform other actions if needed
    #             pass
    def combo_activated(self, index):
        # Assuming self.images is a list of Image instances
        image_instance = self.images[index]
        # Access ft_components through the instance
        if index not in image_instance.ft_components:
            image_instance.Calculations(index)
        selected_combo = image_instance.combos[index].currentText()
        if selected_combo in image_instance.ft_components[index]:
            selected_component = image_instance.ft_components[index][selected_combo]
            # Convert the NumPy array to QPixmap
            q_pixmap = QPixmap.fromImage(QImage(selected_component.data.tobytes(), selected_component.shape[1], selected_component.shape[0], QImage.Format_Grayscale8))
            
            # Set the pixmap to the corresponding ft_image_label
            image_instance.ft_image_label.setPixmap(q_pixmap)



    def open_dialog(self):
        # Create an instance of the custom dialog
        Mixer = MyDialog()

        # Show the dialog
        Mixer.exec_()

        self.output_channels_controlers = {
            'Output 1': {
                'select1 img': '',
                'select2 img': '',
                'slider1 val': 0,
                'slider2 va;': 0,
                'type1': '',
                'type2': ''
            },
            'Output 2': {
                'select1 img': '',
                'select2 img': '',
                'slider1 val': 0,
                'slider2 val': 0,
                'type1': '',
                'type2': ''
            }
        }
        
    
def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
