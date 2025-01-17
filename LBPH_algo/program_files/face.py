import cv2
from imutils import paths
import os
import numpy as np

def detect_faces(net,image,minConfidence=0.5):
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

def load_dataset(input_path,net,minConfidence=0.3,minsamples=15):
    # grab the paths to all images in our input directory, extract all the iamges from that dir
    imagePaths=list(paths.list_images(input_path))
    #grab the name of the image  
    names=[p.split(os.path.sep)[-2] for p in imagePaths]
    (names,counts)=np.unique(names,return_counts=True)
    names=names.tolist()


    faces=[]
    labels=[]

    for imagePath in imagePaths:
        # load the image from disk and extract the name of the person
		# from the subdirectory structure
        image=cv2.imread(imagePath)
        name=imagePath.split(os.path.sep)[-2]

        # only process images that have a sufficient number of
		# examples belonging to the class
        if(counts[names.index(name)]<minsamples):
            continue

        #perfrom the detection
        boxes=detect_faces(net,image,minConfidence)

        #loop over the boxews
        for(startX,startY,endX,endY) in boxes:
            #extract the roi of the face and ocnevrt it inot grey scale
            faceROI=image[startY:endY,startX:endX]
            faceROI=cv2.resize(faceROI,(68,68))
            faceROI=cv2.cvtColor(faceROI,cv2.COLOR_BGR2GRAY)

            faces.append(faceROI)
            labels.append(name)
        
    # convert our faces and labels lists to NumPy arrays
    faces=np.array(faces)
    labels=np.array(labels)

    return(faces,labels)

        



