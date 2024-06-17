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
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier



# Global variables for capturing images
capture_name = ""
capture_count = 0
capture_max_seconds = 0
cap = None
capture_started = False
capture_start_time = 0
capture_frame = None  # Initialize capture_frame globally
image_frame=None
live_feed=False
image_label = None
tk_image = None
random_forest_model = None


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


# # Function to capture images
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
    if not os.path.exists(faces_root):
        os.mkdir(faces_root)
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

def capture_images():
    global capture_name, capture_count, capture_max_seconds, cap, capture_started, capture_start_time
    
    while capture_started:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image from camera.")
            break
        
        image_path = os.path.join(os.getcwd(), f"image_{capture_count}.jpg")
        cv2.imwrite(image_path, frame)

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
    global pca, le, model, random_forest_model
    
    # Delete existing model if it exists
    if os.path.exists('svm_model.pkl'):
        os.remove('svm_model.pkl')
        print("Existing model has been deleted.")

    if os.path.exists('random_forest_model.pkl'):
        os.remove('random_forest_model.pkl')
        print("Existing random model has been deleted.")
    
    # Load data
    print("[INFO] loading dataset...")
    try:
        faces_root = "/home/bmsce/Projects/EigenFaces/Faces"

        (faces, labels) = load_face_dataset(faces_root, net, minConfidence=args["confidence"], minSamples=20)
        
        if len(faces) == 0:
            raise Exception("No faces found in dataset.")

        print("[INFO] {} images in dataset".format(len(faces)))

        # Flatten all 2D faces into a 1D list of pixel intensities
        pcaFaces = np.array([f.flatten() for f in faces])

        # Encode the string labels as integers
        labels = le.fit_transform(labels)

        # Check if there are enough samples to split
        if len(labels) < 2:
            raise Exception("Not enough samples to split into training and testing sets.")

        # Construct our training and testing split
        split = train_test_split(faces, pcaFaces, labels, test_size=0.25, stratify=labels, random_state=42)
        (origTrain, origTest, trainX, testX, trainY, testY) = split

        print("[INFO] creating eigenfaces...")
        start = time.time()
        trainX = pca.fit_transform(trainX)
        end = time.time()
        print("[INFO] computing eigenfaces took {:.4f} seconds".format(end - start)) 

        print("[INFO] training Random Forest classifier And svc ...")

        
        model = SVC(kernel="rbf", C=10.0, gamma=0.001, random_state=42, probability=True)
        model = BaggingClassifier(model, n_estimators=10, random_state=42)
        model.fit(trainX, trainY)

        random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
        random_forest_model.fit(trainX, trainY)

        os.chdir('/home/bmsce/Projects/EigenFaces/Eigenfaces_svm')
        filename = 'random_forest_model.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(random_forest_model, file)
            print(f"Saved the Random Forest model to {filename}")

        filename = 'svm_model.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(random_forest_model, file)
            print(f"Saved the svc to {filename}")
        
        messagebox.showinfo("Training Complete", "Random Forest Model and svc trained successfully!")

    except Exception as e:
        messagebox.showerror("Training Error", f"Error occurred during training:\n{str(e)}")


# Function to predict a single face from an image
def predict_single_face_with_path(image_path):
    global random_forest_model,model, pca, le, net
    
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
        prediction_random_forest = random_forest_model.predict(pca_face)

        # Get name from label encoder
        predicted_name_random = le.inverse_transform(prediction_random_forest)[0]
        predicted_name = le.inverse_transform(prediction)[0]


        # Get the predicted probabilities
        probabilities_random_forest = random_forest_model.predict_proba(pca_face)        
        probabilities = model.predict_proba(pca_face)

        print(f"random forest prediction = {predicted_name_random} and prob = {probabilities_random_forest}")

        return predicted_name_random,probabilities_random_forest,predicted_name, probabilities, (startX, startY, endX, endY)
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
        image_rgb_random = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Check if model is trained
        if model is None:
            messagebox.showerror("Error", "Model not trained yet. Please train the model first.")
            return
        
        # Perform face recognition on the selected image
        predicted_name_random,predicted_proba_random,predicted_name, predicted_proba, box = predict_single_face_with_path(path)
        
        if predicted_name is not None:
            result_text = f"Predicted Name: {predicted_name} and Probability: {np.max(predicted_proba):.2f} for SVC \n Predicted Name: {predicted_name_random} and Probability: {np.max(predicted_proba_random):.2f} for Random forest"
            result_label.config(text=result_text)

            # Draw the bounding box and predicted name on the image
            (startX, startY, endX, endY) = box
            cv2.putText(image_rgb, predicted_name, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(image_rgb, (startX, startY), (endX, endY), (0, 255, 0), 2)

            (startX, startY, endX, endY) = box
            cv2.putText(image_rgb_random, predicted_name_random, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(image_rgb_random, (startX, startY), (endX, endY), (0, 255, 0), 2)
        else:
            result_label.config(text="No face detected or multiple faces detected.")
        
        # Resize the image to fit within the GUI window if necessary
        max_width = 800
        if image_rgb.shape[1] > max_width:
            scale_factor = max_width / image_rgb.shape[1]
            new_height = int(image_rgb.shape[0] * scale_factor)
            image_rgb = cv2.resize(image_rgb, (max_width, new_height))

        max_width = 800
        if image_rgb_random.shape[1] > max_width:
            scale_factor = max_width / image_rgb_random.shape[1]
            new_height = int(image_rgb_random.shape[0] * scale_factor)
            image_rgb_random = cv2.resize(image_rgb_random, (max_width, new_height))

        load_image_onto_image_frame(image_rgb,image_rgb_random)
       
        back_btn.pack(side="top", padx="10", pady="10")

        

# # Function to go back to the home screen
# def go_back():

#     global panel, capture_frame,image_frame,image_label,tk_image,live_feed

#     #Check if capture frame already exists, do nothing if it does
#     if capture_frame:
#         capture_frame.destroy()
#         capture_frame = None  # Reset the global variable
#     if image_frame:
#         image_frame.destroy()
#         image_frame=None
#     if panel:
#         panel.image = None
#     clear_image_frame()

#     # Reset the result label text
#     result_label.config(text=" Welcome to the Home page :) ")

#     # Show essential buttons
#     btn.pack(side="top", padx="10", pady="10")
#     train_btn.pack(side="top", padx="10", pady="10")
#     capture_btn.pack(side="top", padx="10", pady="10")

def go_back():
    global panel, capture_frame, image_frame, image_label, tk_image, live_feed

    # Check if capture frame already exists, do nothing if it does
    if capture_frame:
        capture_frame.destroy()
        capture_frame = None  # Reset the global variable
    if image_frame:
        image_frame.destroy()
        image_frame = None
    if panel:
        panel.image = None
    clear_image_frame()

    # Reset the result label text
    result_label.config(text=" Welcome to the Home page :) ")

    # Pack essential buttons again to ensure the layout remains the same
    btn.pack(side="top", padx="10", pady="10")
    train_btn.pack(side="top", padx="10", pady="10")
    capture_btn.pack(side="top", padx="10", pady="10")
    start_button.pack(side="top", padx="10", pady="10")
    stop_button.pack(side="top", padx="10", pady="10")
    back_btn.pack(side="top", padx="10", pady="10")


def start_capture_gui():
    global panel, capture_frame

    # Check if capture frame already exists, do nothing if it does
    if capture_frame:
        capture_frame.destroy()
        capture_frame = None


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

    count_label = tk.Label(capture_frame, text="Number of Images (min - 15):")
    count_label.pack(anchor="w", padx=10, pady=5)
    count_entry = tk.Entry(capture_frame, width=30)
    count_entry.pack(anchor="w", padx=10, pady=5)

    max_seconds_label = tk.Label(capture_frame, text="Max Seconds:")
    max_seconds_label.pack(anchor="w", padx=10, pady=5)
    max_seconds_entry = tk.Entry(capture_frame, width=30)
    max_seconds_entry.pack(anchor="w", padx=10, pady=5)

    start_capture_btn = tk.Button(capture_frame, text="Start Capture", command=start_capture)
    start_capture_btn.pack(anchor="w", padx=10, pady=10)

    back_btn.pack(side="top", padx="10", pady="10")

def load_image_onto_image_frame(image, image_rgb_random):
    global image_frame, image_label, tk_image, image_label_rgb_random, tk_image_rgb_random

    # Destroy the existing image frame if it exists
    if image_frame:
        image_frame.destroy()
        image_frame = None

    # Create a new image frame
    image_frame = tk.Frame(root, bd=5, relief="solid", borderwidth=2)
    image_frame.pack(side="left", fill="both", padx=10, pady=10)

    # Resize images to fit within the screen dimensions (adjust as needed)
    max_width = root.winfo_screenwidth()
    max_height = root.winfo_screenheight()

    # Resize and convert the first image to a format that Tkinter can use
    pil_image = Image.fromarray(image)
    pil_image.thumbnail((max_width, max_height), Image.LANCZOS)
    tk_image = ImageTk.PhotoImage(pil_image)

    # Resize and convert the second image to a format that Tkinter can use
    pil_image_rgb_random = Image.fromarray(image_rgb_random)
    pil_image_rgb_random.thumbnail((max_width, max_height), Image.LANCZOS)
    tk_image_rgb_random = ImageTk.PhotoImage(pil_image_rgb_random)

    # Get dimensions of the images
    width1, height1 = pil_image.size
    width2, height2 = pil_image_rgb_random.size

    # Dynamically adjust the window width to fit both images side by side
    total_width = width1 + width2 + 300 # 40 for padding and borders
    root.geometry(f"{total_width}x{max(height1, height2) + 100}")  # 100 for padding and borders

    # Create a label to display the first image
    image_label = tk.Label(image_frame, image=tk_image)
    image_label.image = tk_image  # Keep a reference to avoid garbage collection

    # Create a label to display the second image
    image_label_rgb_random = tk.Label(image_frame, image=tk_image_rgb_random)
    image_label_rgb_random.image = tk_image_rgb_random  # Keep a reference to avoid garbage collection

    # Pack the labels into the frame side by side
    image_label.pack(side="left", padx=10, pady=10)
    image_label_rgb_random.pack(side="left", padx=10, pady=10)




def predict_single_face_with_image(image):
    global model, pca, le, net
    
    if model is None:
        print("Error: Model not trained yet.")
        return None, None, None

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

def show_constant_image_live_feed():
    global capture_running, cap

    if not capture_running:
        return

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from camera")
        return

    # Convert the frame to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    predicted_name, predicted_proba, box = predict_single_face_with_image(image_rgb)

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

    
    load_image_onto_image_frame_live_feed(image_rgb)
    
    # Schedule the function to be called again after 10 milliseconds
    root.after(10, show_constant_image_live_feed)


def load_image_onto_image_frame_live_feed(image):
    global image_label, tk_image, image_frame

    # Convert the image to a format that Tkinter can use
    pil_image = Image.fromarray(image)
    tk_image = ImageTk.PhotoImage(pil_image)

    if image_label is None:
        # Create the frame and label if they do not exist
        image_frame = tk.Frame(root, bd=5, relief="solid", borderwidth=2)
        image_frame.pack(side="left", fill="y", padx=10, pady=10)
        image_label = tk.Label(image_frame)
        image_label.pack()

    # Update the image in the label
    image_label.configure(image=tk_image)
    image_label.image = tk_image  # Keep a reference to avoid garbage collection
 
def start_camera_feed():
    global cap, capture_running
    cap = cv2.VideoCapture(0)  # Start video capture from the first camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    capture_running = True
    show_constant_image_live_feed()  # Start the image capture and display loop

def clear_image_frame():
    global image_label, image_frame
    if image_label is not None:
        image_label.destroy()
        image_label = None

    if image_frame is not None:
        image_frame.destroy()
        image_frame = None
    
def stop_camera_feed():
    global capture_running, cap
    capture_running = False
    if cap:
        cap.release()
        cap = None
    cv2.destroyAllWindows()
    clear_image_frame()

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

start_button = tk.Button(button_frame, text="Start Camera", command=start_camera_feed)
start_button.pack(side="top", padx="10", pady="10")

stop_button = tk.Button(button_frame, text="Stop Camera", command=stop_camera_feed)
stop_button.pack(side="top", padx="10", pady="10")

# Create a button to go back to the home screen
back_btn = Button(button_frame, text="Back", command=go_back)
back_btn.pack(side="top", padx="10", pady="10")


# Create a label to display the result
result_label = Label(root, text=" Welcome to the Home page :)  ")
result_label.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

# Start the GUI main loop
root.mainloop()




