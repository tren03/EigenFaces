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
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
args = vars(ap.parse_args())

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

while True:
    ret,frame=cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # cv2.imshow("Face",frame)
    

    boxes=detect_faces(frame)
    for (startX,startY,endX,endY) in boxes:
        cv2.rectangle(frame,(startX,startY),(endX,endY),(0,255,0),4)

    #create a new iamge with a rectangle (image,startcoords,endcoords,color_of_box,thickness)
    cv2.imshow("Face",frame)
    #waits 1 millisec for q to be pressed, if pressed then brek out and clsoe the windows
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
