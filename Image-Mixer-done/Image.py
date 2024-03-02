# from PyQt5 import Qt
from PyQt5.QtWidgets import QSlider,QHBoxLayout , QLabel ,QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage 
from PyQt5 import QtWidgets 
from PyQt5.QtCore import Qt
import numpy as np
import logging
import cv2

logging.basicConfig(filename="Image.log",level=logging.INFO , format='%(levelname)s: %(message)s')



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
        self.combos = combos if combos is not None else []  # Initialize as an empty list if not provided
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

    def Calculations(self):
            if self.image is not None:
                # Convert uint8 array to float64
                image_array_float = self.image.astype(np.float64)
                # ft_image = np.fft.fft2(image_array_float)
                # # Shift zero frequency components to the center
                # ft_image_shifted = np.fft.fftshift(ft_image)
                # # Calculate magnitude, phase, real, and imaginary components
                # self.magnitude_shift = np.abs(ft_image_shifted)
                # self.phase_shift = np.angle(ft_image_shifted)
                # self.real_shift = np.real(ft_image_shifted)
                # self.imaginary_shift = np.imag(ft_image_shifted)
                self.dft = np.fft.fft2(image_array_float)
                self.dft_shift = np.fft.fftshift(self.dft)
                # self.magnitude_shift = (20*np.log(np.abs(self.dft_shift))).astype(np.uint8)
                self.phase_shift = (np.angle(self.dft_shift)).astype(np.uint8)
                # self.real_shift = (20*np.log(np.real(self.dft_shift))).astype(np.uint8)
                self.imaginary_shift = (np.imag(self.dft_shift)).astype(np.uint8)
                epsilon = 1e-10  # Small constant to avoid log(0)

                self.magnitude_shift = (20 * np.log(np.abs(self.dft_shift) + epsilon)).astype(np.uint8)
                self.real_shift = (20 * np.log(np.abs(np.real(self.dft_shift)) + epsilon)).astype(np.uint8)

                self.ft_components = {
                    "FT Magnitude": self.magnitude_shift,
                    "FT Phase": self.phase_shift,
                    "FT Real Component": self.real_shift,
                    "FT Imaginary Component": self.imaginary_shift
                    }

    def check_combo(self, index):
        self.Calculations()
        selected_combo = self.combos[index].currentText()
        if selected_combo in self.ft_components:
            selected_component = self.ft_components[selected_combo]
            for _, value in self.ft_components.items():
                if np.array_equal(value, selected_component):
                    # Convert the NumPy array to QPixmap
                    q_pixmap = QPixmap.fromImage(QImage(value.data.tobytes(), value.shape[1], value.shape[0], QImage.Format_Grayscale8))
                    # Convert QPixmap to NumPy array
                    q_image = q_pixmap.toImage()
                    # Convert QImage to QPixmap and set the display
                    q_pixmap = QPixmap.fromImage(q_image)
                    self.ft_image_label.setPixmap(q_pixmap)


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