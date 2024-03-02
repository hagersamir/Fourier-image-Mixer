# from PyQt5 import Qt
from PyQt5.QtWidgets import QRubberBand,QSlider,QHBoxLayout , QLabel ,QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage ,QPainter, QColor
from PyQt5 import QtWidgets 
from PyQt5.QtCore import  Qt, QSize, QRect, QPointF, QRectF
import numpy as np
import logging
import cv2


logging.basicConfig(filename="Image.log", level=logging.INFO , format='%(levelname)s: %(message)s')



class Image(QtWidgets.QWidget):
    instances =[]
    def __init__(self, image,ft_image, combos, parent=None):
        super().__init__(parent)
        self.image = None
        self.width,self.height = 0,0
        self.image_label = image
        self.ft_components = {}
        self.ft_image_label = ft_image
        self.magnitude_shift = None
        self.phase_shift = None
        self.real_shift = None
        self.imaginary_shift = None
        self.calculated = {}
        self.contrast_coef , self.brightness_coef= 1.0,0.0
        self.combos = combos if combos is not None else []  # Initialize as an empty list if not provided
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self.rubber_band.setGeometry(QRect(0, 0, 0, 0))
        self.rubber_band_showing = False
        self.current_x, self.current_y = 0, 0
        self.selection_rect = QRect()  # Variable to store the selected region rectangle
        self.selection_enabled = False  # Variable to track whether selection is enabled
        self.resizing_handle_size = 8  # Size of the resize handles


        # Append each instance to the class variable
        Image.instances.append(self)


    def Browse(self):
        image_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, 'Open Image File', './', filter="Image File (*.png *.jpg *.jpeg)")
        if image_path:
            # Load the image using cv2
            cv_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if cv_image is not None:
                new_height, new_width = cv_image.shape[:2]
                if self.image is not None and (new_width, new_height) != (self.width, self.height):
                    # Sizes are different, apply adjust_image_sizes function
                    self.adjust_sizes()
                # Update display using cv2 image
                self.update_display(cv_image)
                # Update self.image after loading the first image
                self.image = cv_image
                self.width, self.height = new_width, new_height
                # Adjust sizes after updating the display
                self.adjust_sizes()

    def update_display(self, cv_image):
        if cv_image is not None:
            # Update self.image with the cv_image
            self.image = cv_image
            # Convert cv_image to QPixmap
            height, width = cv_image.shape[:2]
            bytes_per_line = width
            # Create QImage from cv_image
            q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
            # Convert QImage to QPixmap and set the display
            q_pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(q_pixmap)

    def adjust_sizes(self):
        # Check if there are images in the instances list
        valid_images = [image for image in Image.instances if image.image is not None]
        if valid_images:
            # Find the smallest width and height among all images
            min_width = min(image.width for image in valid_images)
            min_height = min(image.height for image in valid_images)
            # Resize images in all instances to the smallest size
            for image in valid_images:
                # Resize using cv2
                resized_image = cv2.resize(image.image, (min_width, min_height))
                # Update the image
                image.update_display(resized_image)
                # Resize the FT component image using QPixmap
                if image.ft_image_label.pixmap() is not None:
                    ft_pixmap = image.ft_image_label.pixmap().scaled(min_width, min_height, Qt.KeepAspectRatio)
                    image.ft_image_label.setPixmap(ft_pixmap)

    def Calculations(self, index):
            if self.image is not None:
                # Convert uint8 array to float64
                image_array_float = self.image.astype(np.float64)
                self.dft = np.fft.fft2(image_array_float)
                self.dft_shift = np.fft.fftshift(self.dft)
                epsilon = 1e-10  # Small constant to avoid log(0)
                self.magnitude_shift = (20 * np.log(np.abs(self.dft_shift) + epsilon)).astype(np.uint8)
                self.phase_shift = (np.angle(self.dft_shift)).astype(np.uint8)
                self.real_shift = (20 * np.log(np.abs(np.real(self.dft_shift)) + epsilon)).astype(np.uint8)
                self.imaginary_shift = (np.imag(self.dft_shift)).astype(np.uint8)
                if index not in self.ft_components:
                    self.ft_components[index] = {}
                self.calculated[index] = True
                self.ft_components[index] = {
                "FT Magnitude": self.magnitude_shift,
                "FT Phase": self.phase_shift,
                "FT Real": self.real_shift,
                "FT Imaginary": self.imaginary_shift
                }
                
                # ft_image = np.fft.fft2(image_array_float)
                # # Shift zero frequency components to the center
                # ft_image_shifted = np.fft.fftshift(ft_image)
                # # Calculate magnitude, phase, real, and imaginary components
                # self.magnitude_shift = np.abs(ft_image_shifted)
                # self.phase_shift = np.angle(ft_image_shifted)
                # self.real_shift = np.real(ft_image_shifted)
                # self.imaginary_shift = np.imag(ft_image_shifted)
                # self.magnitude_shift = (20*np.log(np.abs(self.dft_shift))).astype(np.uint8)
                # self.real_shift = (20*np.log(np.real(self.dft_shift))).astype(np.uint8)

    def check_combo(self, index):
        if index not in self.calculated :
            self.Calculations(index)
        selected_combo = self.combos[index].currentText()
        if selected_combo in self.ft_components[index]:
            selected_component = self.ft_components[index][selected_combo]
            for _ , value in self.ft_components[index].items():
                if np.array_equal(value, selected_component):
                    # Convert the NumPy array to QPixmap
                    q_pixmap = QPixmap.fromImage(QImage(value.data.tobytes(), value.shape[1], value.shape[0], QImage.Format_Grayscale8))
                    # Convert QPixmap to NumPy array
                    q_image = q_pixmap.toImage()
                    # Convert QImage to QPixmap and set the display
                    q_pixmap = QPixmap.fromImage(q_image)
                    self.ft_image_label.setPixmap(q_pixmap)
        
    def calculate_brightness_contrast(self, cv_image):
            result=None
            if cv_image is not None:
                print( self.brightness_coef)
                result = cv2.addWeighted(cv_image, self.contrast_coef, np.zeros_like(cv_image), 0, self.brightness_coef)
                self.image = result
            return result

    def process(self, img_mag, img_phase, img_real, img_imag):
        mag_mask = np.ones_like(img_mag)
        phase_mask = np.zeros_like(img_phase)
        real_mask = np.ones_like(img_real)
        imag_mask = np.zeros_like(img_imag)
        return mag_mask, phase_mask, real_mask, imag_mask
    
    def crop_low_freq(self, mode, img_mag, img_phase, img_real, img_imag):
        magnitude_mask, phase_mask, real_mask, imag_mask = self.process(img_mag, img_phase, img_real, img_imag)
        for h in range(int(self.current_y ), int( self.height -self.current_y)):
            for w in range(int(self.current_x), int( self.width - self.current_x)):
                if mode == 'mag':
                    magnitude_mask[h][w] = img_mag[h][w]
                elif mode == 'phase':
                    phase_mask[h][w] = img_phase[h][w]
                elif mode == 'real':
                    real_mask[h][w] = img_real[h][w]
                elif mode == 'imag':
                    imag_mask[h][w] = img_imag[h][w]
        if mode == 'mag':
            return magnitude_mask
        elif mode == 'phase':
            return phase_mask
        elif mode == 'real':
            return real_mask
        elif mode == 'imag':
            return imag_mask

    def crop_high_freq(self, mode, img_mag, img_phase, img_real, img_imag):
        magnitude_mask, phase_mask, real_mask, imag_mask = self.process(img_mag, img_phase, img_real, img_imag)
        for h in range(int(self.current_y ), int( self.height -self.current_y)):
            for w in range(int(self.current_x), int( self.width - self.current_x)):
                if mode == 'mag':
                    img_mag[h][w] = magnitude_mask[h][w]
                elif mode == 'phase':
                    img_phase[h][w] = phase_mask[h][w]
                elif mode == 'real':
                    img_real[h][w] = real_mask[h][w]
                elif mode == 'imag':
                    img_imag[h][w] = imag_mask[h][w]
        if mode == 'mag':
            return img_mag
        elif mode == 'phase':
            return img_phase
        elif mode == 'real':
            return img_real
        elif mode == 'imag':
            return img_imag
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band_showing = True

    def mouseMoveEvent(self, event):
        if self.rubber_band_showing:
            self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.rubber_band_showing = False
            rect = self.rubber_band.geometry()
            self.current_x, self.current_y = rect.width() / 2, rect.height() / 2
            self.process_selected_region()

    # def paintEvent(self, event):
    #     # Paint the selection rectangle if it exists
    #     if self.selection_rect.isValid():
    #         painter = QPainter(self)
    #         painter.setPen(QColor(0, 0, 255))  # Blue color for the rectangle
    #         painter.drawRect(self.selection_rect)

    def paintEvent(self, event):
        # Paint the selection rectangle if it exists
        if self.selection_rect.isValid():
            # Create a transparent overlay image
            overlay = QImage(self.image.shape[1], self.image.shape[0], QImage.Format_ARGB32)
            overlay.fill(Qt.transparent)
            painter = QPainter(overlay)

            # Draw the semi-transparent blue rectangle on the overlay
            painter.setPen(QColor(0, 0, 255, 128))  # Semi-transparent blue color
            painter.setBrush(QColor(0, 0, 255, 128))
            painter.drawRect(self.selection_rect)

            painter.end()

            # Convert the overlay to QPixmap and draw it on the FT image
            overlay_pixmap = QPixmap.fromImage(overlay)
            ft_pixmap = self.ft_image_label.pixmap()
            ft_pixmap = ft_pixmap.copy() if ft_pixmap else QPixmap(self.image.shape[1], self.image.shape[0])
            painter = QPainter(ft_pixmap)
            painter.drawPixmap(0, 0, overlay_pixmap)
            painter.end()

            # Update the FT image label with the modified QPixmap
            self.ft_image_label.setPixmap(ft_pixmap)

    def process_selected_region(self):
        if self.image is not None:
            for index, combo in enumerate(self.combos):
                selected_combo = combo.currentText()
                if selected_combo in self.ft_components.get(index, {}):
                    selected_component = self.ft_components.get(index, {}).get(selected_combo, None)
                    if selected_component is not None:
                        region = self.get_selected_region(selected_component)
                        # Update the display with the selected region
                        q_pixmap = QPixmap.fromImage(QImage(region.data.tobytes(), region.shape[1], region.shape[0], QImage.Format_Grayscale8))
                        q_image = q_pixmap.toImage()
                        q_pixmap = QPixmap.fromImage(q_image)
                        self.ft_image_label.setPixmap(q_pixmap)

                        # Update the selection rectangle
                        self.selection_rect = QRect(
                            int(self.current_x - region.shape[1] / 2),
                            int(self.current_y - region.shape[0] / 2),
                            region.shape[1],
                            region.shape[0]
                        )

                        # Trigger a repaint to display the selection rectangle
                        self.repaint()


    def get_selected_region(self, component):
        region = np.zeros_like(component)
        h, w = component.shape[:2]
        for i in range(int(self.current_y), int(h - self.current_y)):
            for j in range(int(self.current_x), int(w - self.current_x)):
                region[i, j] = component[i, j]
        return region 