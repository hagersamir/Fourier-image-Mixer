from scipy.fft import fft
import scipy.signal as sig
from scipy import interpolate
from scipy import signal
import numpy as np
import pandas as pd
import pyqtgraph as pg
import os
import time
from scipy.interpolate import interp1d
import pyqtgraph as pg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QGraphicsScene ,QLabel , QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtWidgets, uic 
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from cmath import*
from numpy import *
import sys
import matplotlib
matplotlib.use('Qt5Agg')
from Image import Image as ig
from Image import Image


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load the UI Page
        uic.loadUi(r'task4.ui', self)


        
        image_graphs = [self.image1, self.image2, self.image3, self.image4]
        ft_image_graphs = [self.ft_component1, self.label_33, self.ft_component3, self.ft_component4]
        self.combos = [self.comboBox, self.comboBox_11, self.comboBox_8, self.comboBox_10]
        # Create a list to store Image instances and associated QLabel objects
        self.images = [ig(graph, ft_image, combos=self.combos) for graph, ft_image in zip(image_graphs, ft_image_graphs)]


        #Connections
        # Connect combobox signals to the corresponding check_combo method
        self.comboBox.activated.connect(lambda: self.combo_activated(0))
        self.comboBox_11.activated.connect(lambda: self.combo_activated(1))
        self.comboBox_8.activated.connect(lambda: self.combo_activated(2))
        self.comboBox_10.activated.connect(lambda: self.combo_activated(3))

        # Connect double-click events to each QLabel using a loop
        for label, image_instance in zip(image_graphs, self.images):
            label.mouseDoubleClickEvent = lambda event, instance=image_instance: self.double_click_event(event, instance)

    def double_click_event(self, event, image_instance):
        if event.button() == Qt.LeftButton:
            image_instance.Browse()
    

    def combo_activated(self, index):
        for i, image_instance in enumerate(self.images):
            if i == index:
                # Update the selected combo box
                image_instance.check_combo(index)
            else:
                # Reset other combo boxes or perform other actions if needed
                pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
