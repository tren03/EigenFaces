"""
capture image from video using frame,
take the frame from the video and then detect face with box coordinates ,
add a box in the frame with those coordinates on the frame 

"""
# from LBPH_algo.program_files.face import detect_faces
import cv2
import argparse
import os
import numpy as np
import pickle
from sklearn.decomposition import PCA # Performs Principal Component Analysis
from sklearn.preprocessing import LabelEncoder #  Used to encode the class labels (i.e., names of the individuals) as integers rather than strings
from sklearn.decomposition import PCA # Performs Principal Component Analysis
from sklearn.svm import SVC # we train our support vector machine classifier on the eigen faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.exposure import rescale_intensity # Used to visualize the eigenface representations
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

ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")

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
model = SVC(kernel="rbf", C=15.0, gamma=0.001)
model.fit(trainX, trainY)







def detect_faces(image,minConfidence=0.5):
    #grabbing the dminsions of the image recived from the function
    (h,w)=image.shape[:2]

    blob=cv2.dnn.blobFromImage(image,1.0,(300,300),(104.0,177.0,123.0))

    # pass the blob through the network to obtain the face detections,
	# then initialize a list to store the predicted bounding boxes
    net.setInput(blob)
    detections=net.forward()
    boxes=[]

    for i in range(0,detections.shape[2]):
        #extarct the confidence
        confidence=detections[0,0,i,2]
        if(confidence>minConfidence):
            #ADD THE FACE DETECTED IN THE BOX ka coordinates to the boxes array
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype("int")
            boxes.append((startX,startY,endX,endY))
    return boxes


# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)
#creates a capture object that ca[pptures videos
cap=cv2.VideoCapture(0)
#0 means that it will open the default camera in the system 

#if there isnt a camera
if not cap.isOpened():
    print("no camera")
    exit()


(faces, labels) = load_face_dataset(args["input"], net,
	minConfidence=0.5, minSamples=20)
print("[INFO] {} images in dataset".format(len(faces)))
# flatten all 2D faces into a 1D list of pixel intensities
pcaFaces = np.array([f.flatten() for f in faces])
# encode the string labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

#Opening the camera and calling the detect face func
while True:
    ret,frame=cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # cv2.imshow("Face",frame)
    

    boxes=detect_faces(frame)
    for (startX,startY,endX,endY) in boxes:
        cv2.rectangle(frame,(startX,startY),(endX,endY),(0,255,0),4)
        faceROI = frame[startY:endY, startX:endX] # extract face from image by selecting part of the image 
        faceROI = cv2.resize(faceROI, (47, 62)) # resize face image to 47x62 pixels
        faceROI_gray = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)

        # Apply PCA transformation
        pca_face = pca.transform(faceROI_gray.flatten().reshape(1, -1))

        # Perform face recognition
        prediction = model.predict(pca_face)
        predicted_name = le.inverse_transform(prediction)[0]

        # Draw the recognition result on the frame
        cv2.putText(frame, predicted_name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw the face detection box
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        

        # Draw the recognition result on the frame
        cv2.putText(frame, predicted_name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Draw the face detection box
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    #create a new image with a rectangle (image,startcoords,endcoords,color_of_box,thickness)
    cv2.imshow("Face",frame)
    #waits 1 millisec for q to be pressed, if pressed then brek out and clsoe the windows
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()