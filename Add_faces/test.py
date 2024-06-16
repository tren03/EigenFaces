import tkinter as tk
from PIL import Image, ImageTk
import cv2

def update_image():
    ret, frame = cap.read()
    if ret:
        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to a PIL Image
        image = Image.fromarray(frame_rgb)
        # Convert the PIL Image to a Tkinter PhotoImage
        photo_image = ImageTk.PhotoImage(image)
        # Update the label with the new image
        label.config(image=photo_image)
        label.image = photo_image
        # Call this function again after 1ms (for a ~1000ms delay between frames)
        root.after(1, update_image)

# Open the camera
cap = cv2.VideoCapture(0)

# Initialize tkinter
root = tk.Tk()
root.title("Video Display")

# Create a label to display the image
label = tk.Label(root)
label.pack()

# Start updating the image
update_image()

# Start the tkinter main loop
root.mainloop()

# Release the camera
cap.release()
