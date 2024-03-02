

from scipy.fft import fft
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import  QGraphicsScene
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from cmath import*
from numpy import *
import cv2
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
from Image import Image as ig

class MyDialog(QtWidgets.QDialog):
    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.mode= None
        self.main = main

    def on_changed(self, mode): 
        print('oncahnged called')  
        slider_values = [self.main.verticalSlider.value(), self.main.verticalSlider_3.value(), self.main.verticalSlider_2.value(), self.main.verticalSlider_4.value()]
        self.main.progressBar.setValue(0)
        if self.main.mag_phase_checkbox.isChecked():
            index = 0
            # ["Magnitude", "Phase"]
        else:
            index = 1
            #["Real", "Imaginary"]
        indexes = [combo_box.currentText() for combo_box in self.main.combos]
        self.newimage = self.mix_2(index, *slider_values, indexes, mode)
        cv2.imwrite('test2.jpg', self.newimage )
        self.newimage = cv2.normalize(self.newimage, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        outputgraph = self.main.mixer_output_combobox.currentIndex()
        self.plot_image(np.real(self.newimage), outputgraph)

    def mix_2(self, index, slid1, slid2, slid3, slid4, list_combo_box, mode): 
        newmag, newphase, newreal, newimag = 0, 0, 0, 0
        values = [[], [], [], []]
        Mix_ratios = [slid1 / 100, slid2 / 100, slid3 / 100, slid4 / 100]
        for i in range(4):
            component = list_combo_box[i]
            print(self.main.images[i].instances)
            if self.main.images[i].ft_components:
                values[i] = self.get_component(component, i, mode)
            if self.main.images[i].ft_components:
                if list_combo_box[i] == 'Magnitude':  # Magnitude or Real
                    newmag += Mix_ratios[i] * values[i]
                if list_combo_box[i] == 'Phase': # Phase or Imaginary
                    newphase += Mix_ratios[i] * values[i]
                if list_combo_box[i] == 'Real': 
                    newreal += Mix_ratios[i] * values[i]
                if list_combo_box[i] == 'Imaginary':
                    newimag += Mix_ratios[i] * values[i]
        if index == 0:
            new_mixed_ft = np.multiply(newmag, np.exp(1j * newphase))
        else:
            new_mixed_ft = newreal + 1j * newimag
        now_mixed = self.inverse_fourier(new_mixed_ft)  
        return now_mixed

    def get_component(self, component, img_index, mode):
        if mode == 'nonregion':
            out = self.main.images[img_index].ft_components_mix[component]
        else :
            out = self.main.images[img_index].ft_components_cropped[component]
        return out
        
    def inverse_fourier(self, newimage):
        Inverse_fourier_image = np.real(np.fft.ifft2(np.fft.ifftshift(newimage)))
        return Inverse_fourier_image
    
    def plot_image_on_label(self, image, graph):
        outputgraph = self.main.output_graphs[graph]
        # Get the current QGraphicsScene associated with outputgraph
        current_scene = outputgraph.scene()
        # Check if there is a current scene
        if current_scene:
            # Clear the existing items in the QGraphicsScene
            current_scene.clear()
        else:
            # If no scene exists, create a new QGraphicsScene
            new_scene = QGraphicsScene()
            outputgraph.setScene(new_scene)
        clipped_image_component = np.clip(image, 0, 255).astype(np.uint8)
        image_bytes = clipped_image_component.tobytes()
        height, width = image.shape
        bytes_per_line = width
        q_image = QImage(image_bytes, width, height, bytes_per_line, QImage.Format_Grayscale8)
        # Convert the QImage to a QPixmap
        pixmap = QPixmap.fromImage(q_image)
        # Create a QGraphicsPixmapItem
        pixmap_item = QGraphicsPixmapItem(pixmap)
        # Create a QGraphicsScene
        scene = QGraphicsScene()
        scene.addItem(pixmap_item)
        # Set the QGraphicsScene to the QGraphicsView
        outputgraph.setScene(scene)

    def plot_image(self, image, graph):
            outputgraph = self.main.output_graphs[graph]
            image = cv2.imread(r'test2.jpg')
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            # Convert QImage to QPixmap
            pixmap = QPixmap.fromImage(q_image)
            # Create a QGraphicsPixmapItem
            pixmap_item = QGraphicsPixmapItem(pixmap)
            # Create a QGraphicsScene
            scene = QGraphicsScene()
            scene.addItem(pixmap_item)
            outputgraph.setScene(scene)
            outputgraph.setFixedSize(width, height)
            outputgraph.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            outputgraph.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.main.progressBar.setValue(100)

    def ExtractRegion(self):
            mode = 'region'
            image = self.main.images[0]
            if (image.all_regions):
                x_coor = image.all_regions[0].x()
                y_coor = image.all_regions[0].y()
                height = image.all_regions[0].height()
                width = image.all_regions[0].width()
                for i, image in enumerate(self.main.images):
                    if image.ft_components:
                        self.fshiftcrop = image.dft_shift
                        self.mask = np.zeros_like(self.fshiftcrop)
                        self.mask[int(y_coor):int(y_coor + height),
                            int(x_coor):int(x_coor + width)] = 1
                        # Create a mask with zeros inside rectangle region
                        if self.main.outer_checkbox_1.isChecked():
                            self.fshiftcrop = self.fshiftcrop - self.fshiftcrop * self.mask
                        else:
                            self.fshiftcrop = self.fshiftcrop * self.mask
                        cv2.imwrite('test2.jpg', np.real(np.fft.ifft2(np.fft.ifftshift(self.fshiftcrop))))
                        image.Calculations(i,self.fshiftcrop )
                self.on_changed(mode)
