
Sure, here's the complete README for the Fourier-Image-Mixer project:

Fourier-Image-Mixer
Description:
This desktop program demonstrates the relative importance of magnitude and phase components in a signal by manipulating Fourier transforms of grayscale images. Users can view and manipulate four grayscale images simultaneously, observe their Fourier transform components, and mix them to create new images.

Features:
Image Viewers:
Open and view up to four grayscale images.
Automatic conversion to grayscale for colored images.
Unified sizes for opened images.
View Fourier transform (FT) components: Magnitude, Phase, Real, and Imaginary.
Change images by double-clicking on viewer.
Choose Mixing Mode:
Choose between Mag and phase or Real and Imaginary modes.
Output Ports:
View mixer result in one of two output viewports.
Control which viewport displays the mixer result.
Brightness/Contrast:
Adjust brightness/contrast of images via mouse dragging.
Apply adjustments to any of the four components.
Components Mixer:
Mixer result is the inverse Fourier transform (ifft) of a weighted average of input images' FTs.
Customize weights of each image's FT via sliders.
Intuitive interface for customizing weights for two components.
Regions Mixer:
Select regions (inner or outer) for each FT component.
Choose regions using drawn rectangles with customizable sizes.
Highlight selected regions for visual representation.
Realtime Mixing:
Display progress bar during lengthy ifft operations.
Cancel previous operation when new request is made.
Screenshots:
Main Ui:
![Screenshot (33)](https://github.com/hagersamir/Fourier-image-Mixer/assets/105936147/d31b0ad1-199e-4d04-861a-ab5681d4d5a2)
Adjust Contrast and Brightness:
![Screenshot 2024-03-02 170524](https://github.com/hagersamir/Fourier-image-Mixer/assets/105936147/941c03c9-c984-4c30-90c3-aac9b7ba202b)
Mixing 2 imgs mode1 (mag&phase) in output port 1:
![Screenshot (34)](https://github.com/hagersamir/Fourier-image-Mixer/assets/105936147/466d8b94-183e-45ed-a27f-75821e2a1fed)
Mixing 2 imgs mode2 (real&imaginary) in output port 2:
![Screenshot (35)](https://github.com/hagersamir/Fourier-image-Mixer/assets/105936147/a1c20bac-5016-4f27-838a-d9a25b18d9a1)
Mixing small regions(at center) of 4 imgs mode1 (mag&phase) using inner region in output port 1:
![Screenshot (36)](https://github.com/hagersamir/Fourier-image-Mixer/assets/105936147/2597da73-9001-4ba7-962c-a2c30c7b5865)
Mixing small regions (at center) of 4 imgs mode1 (mag&phase) using outer region in output port 1:
![Screenshot (37)](https://github.com/hagersamir/Fourier-image-Mixer/assets/105936147/898fef9b-c2a7-4934-87b9-8a712cc6cf34)
Mixing large regions (at center) of 4 imgs mode1 (mag&phase) using outer region in output port 1:
![Screenshot (38)](https://github.com/hagersamir/Fourier-image-Mixer/assets/105936147/0e962ade-e1b0-4c7f-b4f5-c8b6eee73885)
Mixing large regions (at center) of 4 imgs mode1 (mag&phase) using inner region in output port 1:
![Screenshot (39)](https://github.com/hagersamir/Fourier-image-Mixer/assets/105936147/1fa35f30-de95-4d34-ab1f-98d73232d6ec)

Instructions:
Installation:

Clone this repository to your local machine.
Install the required dependencies by running pip install -r requirements.txt in your terminal or command prompt.
Navigate to the project directory containing main.py.
Running the Program:

Run the program by executing python main.py in your terminal or command prompt.
The application window will open, displaying the main user interface.
Loading Images:

Click on the "Open Image" button to load up to four grayscale images.
If you load a colored image, the program will automatically convert it to grayscale.
Images will be displayed in separate viewports within the application window.
Adjusting Brightness/Contrast:

Drag the mouse up or down to adjust the brightness of an image.
Drag the mouse left or right to adjust the contrast of an image.
You can adjust the brightness/contrast for each image individually.
Viewing Fourier Transform Components:

Select an image viewport.
Choose the Fourier transform component you want to view (Magnitude, Phase, Real, or Imaginary) from the drop-down menu.
The selected Fourier transform component will be displayed in the corresponding viewport.
Customizing Mixing Settings:

Choose the mixing mode (Mag and phase or Real and Imaginary) from the mode selection buttons.
Use the sliders to customize the weights for each image's Fourier transform components.
Adjust the weights to create the desired mixer result.
Selecting Regions for Mixing:

Click and drag to draw rectangles on the Fourier transform components.
Choose whether to include the inner or outer region of each rectangle for mixing.
Adjust the size of the region rectangles using the slider or resize handles.
Viewing Mixer Result:

Choose one of the two output viewports to display the mixer result.
The mixer result will be displayed in real-time as you adjust the mixing settings.
Realtime Mixing:

During lengthy mixer operations, a progress bar will be displayed to indicate the progress of the operation.
You can cancel the ongoing operation and start a new one by making new requests.
By following these instructions, you can effectively use the Fourier-Image-Mixer application to manipulate Fourier transforms of grayscale images and observe the effects of different mixing settings on the resulting images.

Technologies Used:
Python
PyQt5 for GUI
NumPy for mathematical operations
OpenCV for image processing
Screenshots:
Main Ui:
![Screenshot (33)](https://github.com/hagersamir/Fourier-image-Mixer/assets/105936147/d31b0ad1-199e-4d04-861a-ab5681d4d5a2)
Adjust Contrast and Brightness:
![Screenshot 2024-03-02 170524](https://github.com/hagersamir/Fourier-image-Mixer/assets/105936147/941c03c9-c984-4c30-90c3-aac9b7ba202b)
Mixing 2 imgs mode1 (mag&phase) in output port 1:
![Screenshot (34)](https://github.com/hagersamir/Fourier-image-Mixer/assets/105936147/466d8b94-183e-45ed-a27f-75821e2a1fed)
Mixing 2 imgs mode2 (real&imaginary) in output port 2:
![Screenshot (35)](https://github.com/hagersamir/Fourier-image-Mixer/assets/105936147/a1c20bac-5016-4f27-838a-d9a25b18d9a1)
Mixing small regions(at center) of 4 imgs mode1 (mag&phase) using inner region in output port 1:
![Screenshot (36)](https://github.com/hagersamir/Fourier-image-Mixer/assets/105936147/2597da73-9001-4ba7-962c-a2c30c7b5865)
Mixing small regions (at center) of 4 imgs mode1 (mag&phase) using outer region in output port 1:
![Screenshot (37)](https://github.com/hagersamir/Fourier-image-Mixer/assets/105936147/898fef9b-c2a7-4934-87b9-8a712cc6cf34)
Mixing large regions (at center) of 4 imgs mode1 (mag&phase) using outer region in output port 1:
![Screenshot (38)](https://github.com/hagersamir/Fourier-image-Mixer/assets/105936147/0e962ade-e1b0-4c7f-b4f5-c8b6eee73885)
Mixing large regions (at center) of 4 imgs mode1 (mag&phase) using inner region in output port 1:
![Screenshot (39)](https://github.com/hagersamir/Fourier-image-Mixer/assets/105936147/1fa35f30-de95-4d34-ab1f-98d73232d6ec)

