from imutils import paths
import numpy as np 
import cv2
import os
import albumentations as A
# net is our face detector

def detect_faces(net, image, minConfidence=0.5):
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
    

    # Define augmentation function for different lighting conditions
    def augment_image(image):
        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),  # Adjust brightness and contrast randomly
            # You can add more augmentation techniques here for different lighting conditions
        ])
        augmented = transform(image=image)
        return augmented['image']

    # Rest of the function remains the same
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
            faceROI = image[startY:endY, startX:endX]  # Extract face from image
            faceROI = cv2.resize(faceROI, (47, 62))  # Resize face image
            faceROI_grey = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
            faceROI = np.stack([faceROI_grey] * 3, axis=-1)  # Ensure the image has 3 channels (required by albumentations)
            augmented_face = augment_image(faceROI)  # Apply augmentation for different lighting conditions
            augmented_face = cv2.cvtColor(augmented_face, cv2.COLOR_BGR2GRAY)  # Convert back to grayscale
            
            faces.append(faceROI_grey)
            labels.append(name)
            faces.append(augmented_face)
            labels.append(name)

    faces = np.array(faces)
    labels = np.array(labels)

    return faces, labels
















# # TESTING ##

# inputPath = "../../Faces/vishnu"  # Provide the path to the directory containing the face images
# net = cv2.dnn.readNet('../../Detector_model/deploy.prototxt', '../../Detector_model/res10_300x300_ssd_iter_140000.caffemodel')  # Initialize the face detection neural network
# minConfidence = 0.5  # Minimum confidence threshold for face detection (optional, default is 0.5)
# minSamples = 15  # Minimum number of samples required per face class (optional, default is 15)

# (faces,labels) = load_face_dataset(inputPath,net,minConfidence,minSamples)
# # to display image

# print(labels.size)

# for face, label in zip(faces, labels):
#     img = cv2.imshow(label, face)
#     cv2.waitKey(0)

# cv2.destroyAllWindows() 


