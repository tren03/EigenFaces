# import the necessary packages
from sklearn.preprocessing import LabelEncoder #  Used to encode the class labels (i.e., names of the individuals) as integers rather than strings
from sklearn.decomposition import PCA # Performs Principal Component Analysis
from sklearn.svm import SVC # we train our support vector machine classifier on the eigen faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.exposure import rescale_intensity # Used to visualize the eigenface representations
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from pyimagesearch.faces import load_face_dataset
from pyimagesearch.faces import detect_faces
from imutils import build_montages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import pickle
from imutils import resize
from sklearn.ensemble import BaggingClassifier


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default='../Faces',
	help="path to input directory of images")

# ap.add_argument("-f", "--face", type=str,
# 	default="face_detector",
# 	help="path to face detector model directory")

ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-n", "--num-components", type=int, default=15,
	help="# of principal components")
ap.add_argument("-v", "--visualize", type=int, default=-1,
	help="whether or not PCA components should be visualized")
args = vars(ap.parse_args())

print("[INFO] loading face detector model...")
net = cv2.dnn.readNet('face_detector/deploy.prototxt', 'face_detector/res10_300x300_ssd_iter_140000.caffemodel')


# load the CALTECH faces dataset
print("[INFO] loading dataset...")
(faces, labels) = load_face_dataset(args["input"], net,
	minConfidence=0.5, minSamples=20)
print("[INFO] {} images in dataset".format(len(faces)))
# flatten all 2D faces into a 1D list of pixel intensities
pcaFaces = np.array([f.flatten() for f in faces])
# encode the string labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)
# construct our training and testing split
split = train_test_split(faces, pcaFaces, labels, test_size=0.25,
	stratify=labels, random_state=42)
(origTrain, origTest, trainX, testX, trainY, testY) = split



# compute the PCA (eigenfaces) representation of the data, then
# project the training data onto the eigenfaces subspace
print("[INFO] creating eigenfaces...")
pca = PCA(
	svd_solver="randomized",
	n_components=args["num_components"],
	whiten=True)
start = time.time()
trainX = pca.fit_transform(trainX)
end = time.time()
print("[INFO] computing eigenfaces took {:.4f} seconds".format(
	end - start))



# check to see if the PCA components should be visualized
# if args["visualize"] > 0:
# 	# initialize the list of images in the montage
# 	images = []
# 	# loop over the first 16 individual components
# 	for (i, component) in enumerate(pca.components_[:16]):
# 		# reshape the component to a 2D matrix, then convert the data
# 		# type to an unsigned 8-bit integer so it can be displayed
# 		# with OpenCV
# 		component = component.reshape((62, 47))
# 		component = rescale_intensity(component, out_range=(0, 255))
# 		component = np.dstack([component.astype("uint8")] * 3)
# 		images.append(component)
# 	# construct the montage for the images
# 	montage = build_montages(images, (47, 62), (4, 4))[0]
# 	# show the mean and principal component visualizations
# 	# show the mean image
# 	mean = pca.mean_.reshape((62, 47))
# 	mean = rescale_intensity(mean, out_range=(0, 255)).astype("uint8")
# 	cv2.imshow("Mean", mean)
# 	cv2.imshow("Components", montage)
# 	cv2.waitKey(0)
	


# train a classifier on the eigenfaces representation
print("[INFO] training SVM classifier...")
model = SVC(kernel="rbf", C=10.0, gamma=0.001, random_state=42,probability=True)
model.fit(trainX, trainY)

filename = 'svm_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
print(f"Saved the model to {filename}")


# evaluate the model
print("[INFO] evaluating model...")
predictions = model.predict(pca.transform(testX))
print("*** SVC REPORT ***")
print(classification_report(testY, predictions,	
	target_names=le.classes_))


# # Train a Random Forest classifier on the eigenfaces representation
# print("[INFO] training Random Forest classifier...")
# random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
# random_forest_model.fit(trainX, trainY)
# # Evaluate the model
# print("[INFO] evaluating Random Forest model...")
# predictions_rf = random_forest_model.predict(pca.transform(testX))
# print("*** RANDOM FOREST REPORT ***")
# print(classification_report(testY, predictions_rf, target_names=le.classes_))




# # Train a k-Nearest Neighbors classifier on the eigenfaces representation
# print("[INFO] training k-NN classifier...")
# knn_model = KNeighborsClassifier(n_neighbors=5)
# knn_model.fit(trainX, trainY)
# # Evaluate the model
# print("[INFO] evaluating k-NN model...")
# print("*** KNN REPORT ***")
# predictions_knn = knn_model.predict(pca.transform(testX))
# print(classification_report(testY, predictions_knn, target_names=le.classes_))




# # generate a sample of testing data
# idxs = np.random.choice(range(0, len(testY)), size=2, replace=False)
# # loop over a sample of the testing data
# for i in idxs:
# 	# grab the predicted name and actual name
# 	predName = le.inverse_transform([predictions[i]])[0]
# 	actualName = le.classes_[testY[i]]
# 	# grab the face image and resize it such that we can easily see
# 	# it on our screen
# 	face = np.dstack([origTest[i]] * 3)
# 	face = imutils.resize(face, width=250)
# 	# draw the predicted name and actual name on the image
# 	cv2.putText(face, "pred: {}".format(predName), (5, 25),
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# 	cv2.putText(face, "actual: {}".format(actualName), (5, 60),
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
# 	# display the predicted name  and actual name
# 	print("[INFO] prediction: {}, actual: {}".format(
# 		predName, actualName))
# 	# display the current face to our screen
# 	cv2.imshow("Face", face)
# 	cv2.waitKey(0)




# # Define a function to perform face recognition on a single image
# def predict_single_face(image_path, model, pca, le, net):
#     # Load the input image
#     image = cv2.imread(image_path)

#     # Perform face detection
#     (boxes) = detect_faces(net, image)

#     # Assuming there's only one face in the test image
#     if len(boxes) == 1:
#         # Extract the face ROI
#         (startX, startY, endX, endY) = boxes[0]
#         faceROI = image[startY:endY, startX:endX]

#         # Resize the face ROI
#         faceROI = cv2.resize(faceROI, (47, 62))

#         # Convert the face ROI to grayscale
#         gray_face = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)

#         # Flatten the grayscale face ROI
#         flattened_face = gray_face.flatten()

#         # Perform PCA transformation
#         pca_face = pca.transform(flattened_face.reshape(1, -1))

#         # Perform face recognition prediction
#         prediction = model.predict(pca_face)
#         predicted_name = le.inverse_transform(prediction)[0]

#         # Get the predicted probabilities
#         probabilities = model.predict_proba(pca_face)

#         # Draw the bounding box around the detected face
#         cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        
#         # Display the predicted face and its details
#         face_display = cv2.resize(image, (250, 250))
#         cv2.putText(face_display, f"Pred: {predicted_name}", (5, 25),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#         cv2.putText(face_display, f"Conf: {np.max(probabilities):.2f}", (5, 60),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#         cv2.imshow("Face", face_display)
#         cv2.waitKey(0)

#         return predicted_name, probabilities
#     else:
#         print("Error: Detected more than one face in the test image.")
#         return None, None


# # Test the function on a single image
# image_path = "single_face_test/test2.jpg"
# predicted_name, predicted_proba = predict_single_face(image_path, model, pca, le, net)
# print("Predicted Name:", predicted_name)
# print("Predicted Probability:", predicted_proba)

import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
from pyimagesearch.faces import detect_faces
import pickle
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from skimage.exposure import rescale_intensity
import os

# # Function to train the SVM model on the dataset
def train_model():
    
	# Loadin data - 
	print("[INFO] loading dataset...")
	(faces, labels) = load_face_dataset(args["input"], net,
		minConfidence=0.5, minSamples=20)
	print("[INFO] {} images in dataset".format(len(faces)))
	# flatten all 2D faces into a 1D list of pixel intensities
	pcaFaces = np.array([f.flatten() for f in faces])
	# encode the string labels as integers
	le = LabelEncoder()
	labels = le.fit_transform(labels)
	# construct our training and testing split
	split = train_test_split(faces, pcaFaces, labels, test_size=0.25,
		stratify=labels, random_state=42)
	(origTrain, origTest, trainX, testX, trainY, testY) = split
    

	print("[INFO] creating eigenfaces...")
	pca = PCA(
	svd_solver="randomized",
	n_components=args["num_components"],
	whiten=True)
	start = time.time()
	trainX = pca.fit_transform(trainX)
	end = time.time()
	print("[INFO] computing eigenfaces took {:.4f} seconds".format(
	end - start))
    

	print("[INFO] training SVM classifier...")
	model = SVC(kernel="rbf", C=10.0, gamma=0.001, random_state=42,probability=True)
	model.fit(trainX, trainY)

	filename = 'svm_model.pkl'
	with open(filename, 'wb') as file:
         pickle.dump(model, file)
         print(f"Saved the model to {filename}")

    


# Function to predict a single face from an image
def predict_single_face(image_path, model, pca, le, net):
    # Load the input image
    image = cv2.imread(image_path)

    # Perform face detection
    (boxes) = detect_faces(net, image)

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
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform face recognition on the selected image
        predicted_name, predicted_proba, box = predict_single_face(path, model, pca, le, net)
        
        if predicted_name is not None:
            result_text = f"Predicted Name: {predicted_name}\nProbability: {np.max(predicted_proba):.2f}"
            result_label.config(text=result_text)

            # Draw the bounding box and predicted name on the image
            (startX, startY, endX, endY) = box
            cv2.putText(image, predicted_name, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        else:
            result_label.config(text="No face detected or multiple faces detected.")
        
        # Convert the image to RGB for Tkinter display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = Image.fromarray(image_rgb)
        image_rgb = ImageTk.PhotoImage(image_rgb)

        # If the panel is not None, we need to update it
        if panel is None:
            panel = Label(image=image_rgb)
            panel.image = image_rgb
            panel.pack(side="top", padx=10, pady=10)
        else:
            panel.configure(image=image_rgb)
            panel.image = image_rgb

        btn.pack_forget()
        back_btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

# Function to go back to the home screen
def go_back():
    global panel
    if panel:
        panel.pack_forget()
        panel = None
    result_label.config(text="Predicted Name: \nProbability: ")
    btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
    back_btn.pack_forget()

# Load the saved model using pickle
with open("svm_model.pkl", 'rb') as f:
    model = pickle.load(f)

# Initialize the GUI window
root = tk.Tk()
root.title("Face Recognition")

starting_width = 1200
starting_height = 900
root.geometry(f"{starting_width}x{starting_height}")

# Set the minimum size of the window
root.minsize(starting_width, starting_height)

panel = None

# Create a button to load an image
btn = Button(root, text="Load Image", command=load_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

# Create a button to train the model
train_btn = Button(root, text="Train Model", command=train_model)
train_btn.pack(side="left", padx="10", pady="10")

# Create a button to go back to the home screen
back_btn = Button(root, text="Back", command=go_back)
back_btn.pack_forget()

# Create a label to display the result
result_label = Label(root, text="Predicted Name: \nProbability: ")
result_label.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

# Start the GUI main loop
root.mainloop()
