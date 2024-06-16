import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
import time
from pyimagesearch.faces import detect_faces, load_face_dataset
import pickle
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import argparse

# Global variables for capturing images
capture_name = ""
capture_count = 0
capture_max_seconds = 0
cap = None
capture_started = False
capture_start_time = 0
capture_frame = None  # Initialize capture_frame globally
image_frame=None
image_label=None
tk_image=None
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default='../Faces',
                help="path to input directory of images")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-n", "--num-components", type=int, default=15,
                help="# of principal components")
ap.add_argument("-v", "--visualize", type=int, default=-1,
                help="whether or not PCA components should be visualized")
args = vars(ap.parse_args())

print("[INFO] loading face detector model...")
net = cv2.dnn.readNet('face_detector/deploy.prototxt', 'face_detector/res10_300x300_ssd_iter_140000.caffemodel')

# Global variables for PCA, LabelEncoder, and model
pca = PCA(svd_solver="randomized", n_components=args["num_components"], whiten=True)
le = LabelEncoder()
model = None

# Function to start capturing images
def start_capture():
    global capture_name, capture_count, capture_max_seconds, cap, capture_started, capture_start_time
    
    # Reset capture variables
    capture_name = name_entry.get()
    capture_count = int(count_entry.get())
    capture_max_seconds = int(max_seconds_entry.get())
    capture_started = True
    capture_start_time = time.time()

    # Create directory if not exists
    faces_root = "../Faces"
    os.chdir(faces_root)
    if not os.path.exists(capture_name):
        os.mkdir(capture_name)
    os.chdir(capture_name)

    # Start capturing images
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to open camera.")
        return

    capture_images()

# Function to capture images
def capture_images():
    global capture_name, capture_count, capture_max_seconds, cap, capture_started, capture_start_time, panel
    
    while capture_started:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image from camera.")
            break
        
        image_path = os.path.join(os.getcwd(), f"image_{capture_count}.jpg")
        cv2.imwrite(image_path, frame)

        # Update GUI image display

        # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(image)
        # image = ImageTk.PhotoImage(image)
        
        # if panel is None:
        #     panel = Label(root, image=image)
        #     panel.pack(side="top", padx=10, pady=10, fill=tk.BOTH, expand=True)
        # else:
        #     panel.configure(image=image)
        #     panel.image = image
                # Update the panel with the new image
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb = Image.fromarray(image_rgb)
        image_rgb = ImageTk.PhotoImage(image_rgb)
        
        if panel is None:
            panel = Label(root, image=image_rgb)
            panel.image = image_rgb
            panel.pack(side="top", padx=10, pady=10, fill=tk.BOTH, expand=True)
        else:
            panel.configure(image=image_rgb)
            panel.image = image_rgb

        capture_count -= 1
        if capture_count <= 0 or (time.time() - capture_start_time) >= capture_max_seconds:
            stop_capture()
            break

        time.sleep(1)  # Capture image every second

# Function to stop capturing images
def stop_capture():
    global cap, capture_started
    capture_started = False
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

# Function to train the SVM model on the dataset
def train_model():
    global pca, le, model
    
    # Load data
    print("[INFO] loading dataset...")
    try:
        print(args["input"])
        (faces, labels) = load_face_dataset(args["input"], net,
                                            minConfidence=args["confidence"], minSamples=20)
        print("[INFO] {} images in dataset".format(len(faces)))

        # Flatten all 2D faces into a 1D list of pixel intensities
        pcaFaces = np.array([f.flatten() for f in faces])

        # Encode the string labels as integers
        labels = le.fit_transform(labels)

        # Construct our training and testing split
        split = train_test_split(faces, pcaFaces, labels, test_size=0.25,
                                stratify=labels, random_state=42)
        (origTrain, origTest, trainX, testX, trainY, testY) = split

        print("[INFO] creating eigenfaces...")
        start = time.time()
        trainX = pca.fit_transform(trainX)
        end = time.time()
        print("[INFO] computing eigenfaces took {:.4f} seconds".format(end - start))

        print("[INFO] training SVM classifier...")
        global model
        model = SVC(kernel="rbf", C=10.0, gamma=0.001, random_state=42, probability=True)
        model.fit(trainX, trainY)

        filename = 'svm_model.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
            print(f"Saved the model to {filename}")
        
        messagebox.showinfo("Training Complete", "SVM Model trained successfully!")

    except Exception as e:
        messagebox.showerror("Training Error", f"Error occurred during training:\n{str(e)}")

# Function to predict a single face from an image
def predict_single_face(image_path):
    global model, pca, le, net
    
    if model is None:
        print("Error: Model not trained yet.")
        return None, None, None

    # Load the input image
    image = cv2.imread(image_path)

    # Perform face detection
    boxes = detect_faces(net, image)

    # Assuming there's only one face in the test image
    if len(boxes) == 1:
        # Extract the face ROI
        (startX, startY, endX, endY) = boxes[0]
        faceROI = image[startY:endY, startX:endX]

        # Resize the face ROI
        faceROI = cv2.resize(faceROI, (47, 62))

        # Convert the face ROI to grayscale
        gray_face = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)

        # Flatten the grayscale face ROI
        flattened_face = gray_face.flatten()

        # Perform PCA transformation
        pca_face = pca.transform(flattened_face.reshape(1, -1))

        # Perform face recognition prediction
        prediction = model.predict(pca_face)
        predicted_name = le.inverse_transform(prediction)[0]

        # Get the predicted probabilities
        probabilities = model.predict_proba(pca_face)

        return predicted_name, probabilities, (startX, startY, endX, endY)
    else:
        print("Error: Detected more than one face in the test image.")
        return None, None, None

# Function to load an image and perform face recognition
def load_image():
    global panel

    # Open a file dialog to select an image
    path = filedialog.askopenfilename()

    if len(path) > 0:
        # Load the image and convert it to a format that Tkinter can display
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Error", "Failed to load image.")
            return
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Check if model is trained
        if model is None:
            messagebox.showerror("Error", "Model not trained yet. Please train the model first.")
            return
        
        # Perform face recognition on the selected image
        predicted_name, predicted_proba, box = predict_single_face(path)
        
        if predicted_name is not None:
            result_text = f"Predicted Name: {predicted_name}\nProbability: {np.max(predicted_proba):.2f}"
            result_label.config(text=result_text)

            # Draw the bounding box and predicted name on the image
            (startX, startY, endX, endY) = box
            cv2.putText(image_rgb, predicted_name, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(image_rgb, (startX, startY), (endX, endY), (0, 255, 0), 2)
        else:
            result_label.config(text="No face detected or multiple faces detected.")
        
        # Resize the image to fit within the GUI window if necessary
        max_width = 800
        if image_rgb.shape[1] > max_width:
            scale_factor = max_width / image_rgb.shape[1]
            new_height = int(image_rgb.shape[0] * scale_factor)
            image_rgb = cv2.resize(image_rgb, (max_width, new_height))
        
        # Convert the image to RGB for Tkinter display
        # image_rgb = Image.fromarray(image_rgb)
        # image_rgb = ImageTk.PhotoImage(image_rgb)

        load_image_onto_image_frame(image_rgb)
        # Update the panel with the new image
        # if panel is None:
        #     panel = Label(image=image_rgb)
        #     panel.image = image_rgb
        #     panel.pack(side="top", padx=10, pady=10, fill=tk.BOTH, expand=True)
        # else:
        #     panel.config(image=image_rgb)
        #     panel.image = image_rgb

        # Show the back button
        back_btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
        
        # Hide the other buttons if necessary
        btn.pack_forget()

# Function to go back to the home screen
def go_back():
    global panel, capture_frame,image_frame

    # Check if capture frame already exists, do nothing if it does
    # if capture_frame:
    #     capture_frame.destroy()
    #     capture_frame = None  # Reset the global variable

    if panel:
        panel.image = None

    # Reset the result label text
    result_label.config(text=" Welcome to the Home page :) ")

    # Show essential buttons
    btn.pack(side="top", padx="10", pady="10")
    train_btn.pack(side="top", padx="10", pady="10")
    capture_btn.pack(side="top", padx="10", pady="10")

    # Hide back button
    back_btn.pack_forget()

    # Forget the capture details entry widgets
    # name_entry.pack_forget()
    # count_entry.pack_forget()
    # max_seconds_entry.pack_forget()
    # name_label.pack_forget()
    # count_label.pack_forget()
    # max_seconds_label.pack_forget()
    # start_capture_btn.pack_forget()
    if capture_frame:
        capture_frame.pack_forget()
    if image_frame:
        image_frame.pack_forget()

def start_capture_gui():
    global panel, capture_frame

    # Check if capture frame already exists, do nothing if it does
    if capture_frame:
        capture_frame.destroy()
        capture_frame = None  # Reset the global variable
        
    result_label.config(text=" Enter details for capturing images ")

    # Create a new frame for capturing images
    capture_frame = tk.Frame(root,bd=5, relief="solid", borderwidth=2)
    capture_frame.pack(side="left", fill="y", padx=10, pady=10)

    # Capture details entry
    global name_entry, count_entry, max_seconds_entry, name_label, count_label, max_seconds_label, start_capture_btn
    name_label = tk.Label(capture_frame, text="Name:")
    name_label.pack(anchor="w", padx=10, pady=5)
    name_entry = tk.Entry(capture_frame, width=30)
    name_entry.pack(anchor="w", padx=10, pady=5)

    count_label = tk.Label(capture_frame, text="Number of Images:")
    count_label.pack(anchor="w", padx=10, pady=5)
    count_entry = tk.Entry(capture_frame, width=30)
    count_entry.pack(anchor="w", padx=10, pady=5)

    max_seconds_label = tk.Label(capture_frame, text="Max Seconds:")
    max_seconds_label.pack(anchor="w", padx=10, pady=5)
    max_seconds_entry = tk.Entry(capture_frame, width=30)
    max_seconds_entry.pack(anchor="w", padx=10, pady=5)

    start_capture_btn = tk.Button(capture_frame, text="Start Capture", command=start_capture)
    start_capture_btn.pack(anchor="w", padx=10, pady=10)

    back_btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

def load_image_onto_image_frame(image):
    global image_frame, image_label, tk_image

    # If image frame doesn't exist, create it
    if image_frame:
        image_frame.destroy()
        image_frame = None  # Reset the global variable
    

    image_frame = tk.Frame(root, bd=5, relief="solid", borderwidth=2)
    image_frame.pack(side="left", fill="y", padx=10, pady=10)

    # Convert the image to a format that Tkinter can use
    pil_image = Image.fromarray(image)

    tk_image = ImageTk.PhotoImage(pil_image)

    # If image label already exists, update its image

    # Create a label to display the image
    image_label = tk.Label(image_frame, image=tk_image)
    image_label.image = tk_image  # Keep a reference to avoid garbage collection

    # Pack the label into the frame
    image_label.pack()
    # Keep a reference to the image to avoid garbage collection
# Initialize the GUI window
root = tk.Tk()
root.title("Face Recognition")

starting_width = 1200
starting_height = 900
root.geometry(f"{starting_width}x{starting_height}")

# Set the minimum size of the window
root.minsize(starting_width, starting_height)

panel = None

# Create a frame for the buttons on the right side
button_frame = tk.Frame(root,bd=5, relief="solid", borderwidth=2)
button_frame.pack(side="right", fill="y", padx=10, pady=10)

# Create a button to load an image
btn = Button(button_frame, text="Load Image", command=load_image)
btn.pack(side="top", padx="10", pady="10")

# Create a button to train the model
train_btn = Button(button_frame, text="Train Model", command=train_model)
train_btn.pack(side="top", padx="10", pady="10")

# Create a button to start capturing images
capture_btn = Button(button_frame, text="Get New Images", command=start_capture_gui)
capture_btn.pack(side="top", padx="10", pady="10")

# Create a button to go back to the home screen
back_btn = Button(button_frame, text="Back", command=go_back)

# Create a label to display the result
result_label = Label(root, text=" Welcome to the Home page :)  ")
result_label.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

# Start the GUI main loop
root.mainloop()

