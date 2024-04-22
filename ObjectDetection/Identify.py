import tkinter as tk
from gestures_dataset import fist, open_hand, peace_sign, thumbs_up
import cv2
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image, ImageTk

num_classes = 4
window = tk.Tk()

train_data = [fist, open_hand, peace_sign, thumbs_up]

class OpenCamera:
    def __init__(self, window, gesture_predictor):
        self.window = window
        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
            print("Unable to open camera")
            return
        print("Camera opened successfully")
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()
        self.btn = tk.Button(window, text='Shot!', command=self.snapshot)
        self.btn.pack()
        self.gesture_predictor = gesture_predictor
        self.label = tk.Label(window, text="Predicted: ")
        self.label.pack()
        self.update_camera()

    def update_camera(self):
        ret, frame = self.vid.read()
        if ret:
            self.photo = self.convert_to_tkimage(frame)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(10, self.update_camera)  # Update every 10 milliseconds

    def snapshot(self):
        ret, frame = self.vid.read()
        if ret:
        # Resize the frame to match the expected input shape of the model
            frame_resized = cv2.resize(frame, (224, 224))  # Assuming the model expects input size of (224, 224)
            
            frame_tensor = transforms.ToTensor()(frame_resized)  # Convert frame to tensor
            frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension
            
            predicted = self.gesture_predictor(frame_tensor)

            _, predicted_class = torch.max(predicted, 1)
            gesture_labels = ['fist', 'open_hand', 'peace_sign', 'thumbs_up']
            predicted_label = gesture_labels[predicted_class.item()]

            self.label.config(text=f"Predicted: {predicted_label}")



    def convert_to_tkimage(self, frame):
        b, g, r = cv2.split(frame)
        img = cv2.merge((r, g, b))
        pil_img = Image.fromarray(img)
        return ImageTk.PhotoImage(image=pil_img)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

class GesturePredict(nn.Module):
    def __init__(self):
        super(GesturePredict, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(-1, 32 * 56 * 56)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

gp = GesturePredict()
op = OpenCamera(window, gp)

window.mainloop()
