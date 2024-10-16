Real-Time Facial Emotion Detection System
This project is a real-time facial emotion detection system using Python, OpenCV, and Keras. It detects and classifies emotions from both image uploads and live video feeds using a pre-trained model.

Features
Predict emotions from uploaded images
Predict emotions from live video using webcam
GUI created with Tkinter for easy user interaction

Requirements
Ensure you have the following libraries installed before running the project:

Python Version
Python 3.x

Libraries
tkinter: For creating the graphical user interface (comes pre-installed with Python on most systems)
opencv-python: For image and video processing
opencv-python-headless: For running OpenCV without GUI features (optional, if you are using a headless server)
numpy: For array manipulation and mathematical operations
Pillow: For image handling in the GUI
keras: For loading and running the pre-trained emotion detection model
tensorflow: Backend for Keras
matplotlib: (optional) If you plan to plot any data in the future

Installation
Install the required libraries using pip:
pip install opencv-python opencv-python-headless numpy Pillow keras tensorflow matplotlib

Additional Files
Pre-trained Model: Download or train your emotion detection model and save it as model_file_30epochs.h5. Place it in the project directory.
Haar Cascade for Face Detection: Ensure you have the Haar Cascade XML file for face detection (haarcascade_frontalface_default.xml). You can download it from here and save it in the project directory.

Running the Application
Clone the repository or download the project files.
Ensure that the pre-trained model (model_file_30epochs.h5) and Haar Cascade file (haarcascade_frontalface_default.xml) are in the project directory.
Run the main.py file:

python main.py

Using the Application
Predict Image: Click on the "PREDICT IMAGE" button and upload an image to predict emotions.
Predict Face from Live Video: Click on the "PREDICT FACE FROM LIVE VIDEO" button to start the webcam and detect facial emotions in real-time.
To exit, press the 'q' key while using live video prediction or close the application window.
