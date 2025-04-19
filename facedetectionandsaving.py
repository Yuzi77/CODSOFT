import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk

# Load OpenCV's Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection (No face_recognition)")
        self.video_frame = tk.Label(root)
        self.video_frame.pack()
        self.capture = cv2.VideoCapture(0)
        self.update_frame()

        self.add_btn = tk.Button(root, text="Capture Face", command=self.add_new_face)
        self.add_btn.pack(pady=10)

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Convert to ImageTk
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)
        self.root.after(10, self.update_frame)

    def add_new_face(self):
        name = simpledialog.askstring("Name", "Enter the person's name:")
        if not name:
            return
        os.makedirs(f"known_faces/{name}", exist_ok=True)
        ret, frame = self.capture.read()
        if ret:
            path = f"known_faces/{name}/{name}_{len(os.listdir(f'known_faces/{name}'))+1}.jpg"
            cv2.imwrite(path, frame)
            print(f"Saved face image to {path}")

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()
