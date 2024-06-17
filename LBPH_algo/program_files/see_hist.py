import cv2
import numpy as np
import matplotlib.pyplot as plt
from lbph import train_lbph
from face import detect_faces
import argparse
import os
# Load the image in grayscale
ap = argparse.ArgumentParser()

ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

faces=[]
image_path = '/home/sriram/Desktop/programfiles/programmingfiles/EigenFaces/Faces/sriram/image_3.jpg'  # Replace with your image path
image = cv2.imread(image_path)

boxes=detect_faces(net,image)

        #loop over the boxews
for(startX,startY,endX,endY) in boxes:
        #extract the roi of the face and ocnevrt it inot grey scale
        faceROI=image[startY:endY,startX:endX]
        faceROI=cv2.resize(faceROI,(68,68))
        faceROI=cv2.cvtColor(faceROI,cv2.COLOR_BGR2GRAY)
        faces.append(faceROI)

faces=np.array(faces)

# Train the image to get 
histograms = train_lbph(faces)

# Plot the original image
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Plot the histogram
plt.subplot(1, 2, 2)
plt.plot(histograms[0])  # Assuming the first histogram is for our image
plt.title('LBP Histogram')
plt.xlabel('Bins')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
