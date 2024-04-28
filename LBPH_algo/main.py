from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from program_files.face import load_dataset
from program_files import lbph as lb

#arguments list 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
	help="path to input directory of images")
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

# load the CALTECH faces dataset
print("[INFO] loading dataset...")
(faces, labels) = load_dataset(args["input"], net,
	minConfidence=0.5, minsamples=20)
print("[INFO] {} images in dataset".format(len(faces)))

le=LabelEncoder()
labels=le.fit_transform(labels)

(trainX,testX,trainY,testY)=train_test_split(faces,labels,test_size=0.25,stratify=labels,random_state=42)

# trian the algo


# print("LBPH ALGO TIME")
# recogniser=cv2.face.LBPHFaceRecogniser_create(radius=2,neighbours=6,grid_x=8,grid_y=8)
# start=time.time()
# recogniser.train(trainX,trainY)
# end=time.time()
# print(f"info training took{(end-start)} ")

print("LBPH ALGO TIME")
start=time.time()
recognizer=lb.train_lbph(trainX)
print('done')
np.save('trainedRec.npy',recognizer)
end=time.time()
print(f"info training took{(end-start)} ")



print("info gathering time")
predictions=[]
confidence=[]
start=time.time()

trained_face_recognizer=np.load('trainedRec.npy')
for i in range(0,len(testX)):
    (prediction,conf)=lb.predict_lbph(testX[i],trained_face_recognizer,trainY)
    predictions.append(prediction)
    confidence.append(conf)

end=time.time()
print(f"the time taken for predtion is {end-start} ")

print(classification_report(testY,predictions,target_names=le.classes_))

idxs = np.random.choice(range(0, len(testY)), size=10, replace=False)
# loop over a sample of the testing data
for i in idxs:
	# grab the predicted name and actual name
	predName = le.inverse_transform([predictions[i]])[0]
	actualName = le.classes_[testY[i]]
	# grab the face image and resize it such that we can easily see
	# it on our screen
	face = np.dstack([testX[i]] * 3)
	face = imutils.resize(face, width=250)
	# draw the predicted name and actual name on the image
	cv2.putText(face, "pred: {}".format(predName), (5, 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
	cv2.putText(face, "actual: {}".format(actualName), (5, 60),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
	# display the predicted name, actual name, and confidence of the
	# prediction (i.e., chi-squared distance; the *lower* the distance
	# is the *more confident* the prediction is)
	print("[INFO] prediction: {}, actual: {}, confidence: {:.2f}".format(
		predName, actualName, confidence[i]))
	# display the current face to our screen
	cv2.imshow("Face", face)
	cv2.waitKey(0)