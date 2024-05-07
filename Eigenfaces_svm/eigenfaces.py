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
from imutils import build_montages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import pickle

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
if args["visualize"] > 0:
	# initialize the list of images in the montage
	images = []
	# loop over the first 16 individual components
	for (i, component) in enumerate(pca.components_[:16]):
		# reshape the component to a 2D matrix, then convert the data
		# type to an unsigned 8-bit integer so it can be displayed
		# with OpenCV
		component = component.reshape((62, 47))
		component = rescale_intensity(component, out_range=(0, 255))
		component = np.dstack([component.astype("uint8")] * 3)
		images.append(component)
	# construct the montage for the images
	montage = build_montages(images, (47, 62), (4, 4))[0]
	# show the mean and principal component visualizations
	# show the mean image
	mean = pca.mean_.reshape((62, 47))
	mean = rescale_intensity(mean, out_range=(0, 255)).astype("uint8")
	cv2.imshow("Mean", mean)
	cv2.imshow("Components", montage)
	cv2.waitKey(0)
	


# train a classifier on the eigenfaces representation
print("[INFO] training SVM classifier...")
model = SVC(kernel="rbf", C=10.0, gamma=0.001, random_state=42)
model.fit(trainX, trainY)

filename = "final_svc_model.sav"
pickle.dump(model,open(filename,'wb'))

# evaluate the model
print("[INFO] evaluating model...")
predictions = model.predict(pca.transform(testX))
print("*** SVC REPORT ***")
print(classification_report(testY, predictions,
	target_names=le.classes_))



# Train a Random Forest classifier on the eigenfaces representation
print("[INFO] training Random Forest classifier...")
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(trainX, trainY)
# Evaluate the model
print("[INFO] evaluating Random Forest model...")
predictions_rf = random_forest_model.predict(pca.transform(testX))
print("*** RANDOM FOREST REPORT ***")
print(classification_report(testY, predictions_rf, target_names=le.classes_))




# Train a k-Nearest Neighbors classifier on the eigenfaces representation
print("[INFO] training k-NN classifier...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(trainX, trainY)
# Evaluate the model
print("[INFO] evaluating k-NN model...")
print("*** KNN REPORT ***")
predictions_knn = knn_model.predict(pca.transform(testX))
print(classification_report(testY, predictions_knn, target_names=le.classes_))





# generate a sample of testing data
idxs = np.random.choice(range(0, len(testY)), size=22, replace=False)
# loop over a sample of the testing data
for i in idxs:
	# grab the predicted name and actual name
	predName = le.inverse_transform([predictions[i]])[0]
	actualName = le.classes_[testY[i]]
	# grab the face image and resize it such that we can easily see
	# it on our screen
	face = np.dstack([origTest[i]] * 3)
	face = imutils.resize(face, width=250)
	# draw the predicted name and actual name on the image
	cv2.putText(face, "pred: {}".format(predName), (5, 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
	cv2.putText(face, "actual: {}".format(actualName), (5, 60),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
	# display the predicted name  and actual name
	print("[INFO] prediction: {}, actual: {}".format(
		predName, actualName))
	# display the current face to our screen
	cv2.imshow("Face", face)
	cv2.waitKey(0)


