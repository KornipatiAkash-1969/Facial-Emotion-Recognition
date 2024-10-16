import tkinter as tk
from tkinter import messagebox, filedialog, Canvas
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model

def predict_emotion(image_path):
    frame = cv2.imread(image_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    
    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def predict_live_video():
    video = cv2.VideoCapture(0)

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in faces:
            sub_face_img = gray[y:y+h, x:x+w]
            resized = cv2.resize(sub_face_img, (48, 48))
            normalize = resized / 255.0
            reshaped = np.reshape(normalize, (1, 48, 48, 1))
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()
        # Release video capture resources
        cv2.destroyAllWindows()

window = tk.Tk()
window.title("REAL-TIME FACIAL EMOTION DETECTION SYSTEM")
window.geometry("600x700")
window.resizable(0, 0)

load1 = Image.open("1.png")
photo1 = ImageTk.PhotoImage(load1)

header = tk.Button(window, image=photo1, bd=0)  # Set border width to 0 to remove button border
header.place(x=5, y=0)

load2 = Image.open("2.png")
photo2 = ImageTk.PhotoImage(load2)

canvas1 = tk.Canvas(window, width=900, height=250)
canvas1.place(x=5, y=140)
canvas1.create_image(250, 125, image=photo2)

model = load_model('model_file_30epochs.h5')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

def upload_image():
    filename = filedialog.askopenfilename()
    if filename:
        predict_emotion(filename)

b1 = tk.Button(canvas1, text="PREDICT IMAGE", font=("Algerian", 19), bg="gray", fg="black", bd=0, command=upload_image)
b1.place(x=5, y=50)
b1.configure(borderwidth=5)  

b2 = tk.Button(canvas1, text="PREDICT FACE FROM LIVE VIDEO", font=("Algerian", 19), bg="gray", fg="black", bd=0, command=predict_live_video)  # Set border width to 0 to remove button border
b2.place(x=5, y=150)
b2.configure(borderwidth=5)

window.geometry("725x405")
window.protocol("WM_DELETE_WINDOW", on_closing)
window.mainloop()


