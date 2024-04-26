from imutils import paths
import numpy as np 
import cv2
import os

# net is our face detector

def detect_faces(net, image, minConfidence=0.5):
    # grab the dimensions of the image and then construct a blob from it
    
# The cv2.dnn.blobFromImage() function is used to preprocess an image before feeding it into a deep neural network (DNN) for tasks like object detection, classification, or segmentation.
# Here's what each parameter does:
# image: This is the input image that you want to process.
# scalefactor: This parameter is used to scale down (or up) the image before feeding it into the neural network. In this case, it's set to 1.0, which means the image is not scaled.
# size: This parameter specifies the spatial size that the input image will be resized to before passing it through the network. It's given as a tuple (width, height). In this case, the size is set to (300, 300), meaning the image will be resized to a width of 300 pixels and a height of 300 pixels.
# mean: This parameter is the mean subtraction values. It's a tuple in BGR order that's subtracted from each channel of the image. It's used for data normalization. These values are typically obtained from the training dataset. In this case, (104.0, 177.0, 123.0) are the mean values.
# The function returns a 4-dimensional NumPy array (a "blob") representing the preprocessed image that can be passed to the neural network for inference. This array typically has the shape (batch_size, num_channels, width, height), where:
# batch_size: The number of images processed at once. In this case, it's 1.
# num_channels: The number of channels in the image. Usually 3 for RGB images.
# width: The width of the image after resizing.
# height: The height of the image after resizing.

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))
    
    net.setInput(blob) # This line sets the input data (blob) to be processed by the neural network (net). 
    detections = net.forward() # This line performs forward pass inference using the neural network (net). The forward() method executes a single forward pass of the input data through the network and computes the output.
    boxes = [] # will be used to store coordingtes of bounding box


    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > minConfidence:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # update our bounding box results list
            boxes.append((startX, startY, endX, endY))
    # return the face detection bounding boxes
    return boxes


def load_face_dataset(inputPath, net, minConfidence=0.5, minSamples=15):
    
	#This line uses the list_images function from the paths module of the imutils library to obtain a list of file paths to all the images in the inputPath directory. These paths are stored in the imagePaths list.
    imagePaths = list(paths.list_images(inputPath))
    
	#if image path = faces_organized/p1/img_001, p.split(os.path.sep) converts it into a list seperating based on the os.path.sep which is a seperator used in file paths. It would return a list ["faces_organized,","p1","img_001"]. The [-2] indicates we extrace 2nd index from the ending of the array and store it in names array
    names = [p.split(os.path.sep)[-2] for p in imagePaths]


	#returns the count of unique names present which is the nubmer of images for each person
    (names, counts) = np.unique(names, return_counts=True)
    
	#convert names to list as it is a ndarray
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
            faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)

            faces.append(faceROI)
            labels.append(name)

    faces = np.array(faces)
    labels = np.array(labels)

    return (faces, labels)



