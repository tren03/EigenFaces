import cv2
import numpy as np
import argparse
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imutils import paths
import albumentations as A

# Argument parser setup
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default='../Faces', help="path to input directory of images")
ap.add_argument("-f", "--face", type=str, default="face_detector", help="path to face detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-n", "--num-components", type=int, default=150, help="# of principal components")
args = vars(ap.parse_args())

print("[INFO] loading face detector model...")
net = cv2.dnn.readNet('face_detector/deploy.prototxt', 'face_detector/res10_300x300_ssd_iter_140000.caffemodel')

def augment_image(image):
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5)
    ])
    augmented = transform(image=image)
    return augmented['image']

def detect_faces(net, image, minConfidence=0.5):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > minConfidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            boxes.append((startX, startY, endX, endY))
    return boxes

def load_face_dataset(inputPath, net, minConfidence=0.5, minSamples=20):
    imagePaths = list(paths.list_images(inputPath))
    names = [p.split(os.path.sep)[-2] for p in imagePaths]
    (names, counts) = np.unique(names, return_counts=True)
    names = names.tolist()

    faces = []
    labels = []
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        name = imagePath.split(os.path.sep)[-2]

        if counts[names.index(name)] < minSamples:
            continue

        boxes = detect_faces(net, image, minConfidence)
        for (startX, startY, endX, endY) in boxes:
            faceROI = image[startY:endY, startX:endX]
            faceROI = cv2.resize(faceROI, (47, 62))
            faceROI_gray = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)
            faces.append(faceROI_gray)
            labels.append(name)
            augmented_face = augment_image(faceROI)
            augmented_face_gray = cv2.cvtColor(augmented_face, cv2.COLOR_BGR2GRAY)
            faces.append(augmented_face_gray)
            labels.append(name)

    faces = np.array(faces)
    labels = np.array(labels)
    return faces, labels

# Load dataset
print("[INFO] loading dataset...")
faces, labels = load_face_dataset(args["input"], net, minConfidence=args["confidence"])
print(f"[INFO] {len(faces)} images in dataset")

# Flatten faces and encode labels
faces = np.array([f.flatten() for f in faces])
le = LabelEncoder()
labels = le.fit_transform(labels)

# Split the data
trainX, testX, trainY, testY = train_test_split(faces, labels, test_size=0.25, stratify=labels, random_state=42)

# Normalize the data
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)

# Apply PCA
print("[INFO] applying PCA...")
pca = PCA(n_components=args["num_components"], whiten=True)
trainX = pca.fit_transform(trainX)
testX = pca.transform(testX)

# Train the SVM model
print("[INFO] training SVM classifier...")
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1)
grid.fit(trainX, trainY)

print("[INFO] best hyperparameters: ", grid.best_params_)
model = grid.best_estimator_

# Evaluate the model
print("[INFO] evaluating the model...")
predictions = model.predict(testX)
print(classification_report(testY, predictions, target_names=le.classes_))

# Live face recognition
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("no camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    boxes = detect_faces(net, frame)
    for (startX, startY, endX, endY) in boxes:
        faceROI = frame[startY:endY, startX:endX]
        faceROI = cv2.resize(faceROI, (47, 62))
        faceROI_gray = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)
        face_flat = faceROI_gray.flatten().reshape(1, -1)
        face_scaled = scaler.transform(face_flat)
        face_pca = pca.transform(face_scaled)
        prediction = model.predict(face_pca)
        predicted_name = le.inverse_transform(prediction)[0]

        cv2.putText(frame, predicted_name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("Face", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
