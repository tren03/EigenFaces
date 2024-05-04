#a python gui which does the following 

"""asks the user for name from terminal and also count of images ass count , takes pictures normally every sec for until count has been made max seconds wh
"""

import cv2 
import time
import os

cwd = os.getcwd()

# Get all the directory names in the current working directory
os.chdir("Faces")
# import cv2.qt

# import cv2.qt.plugins

# import cv2.qt.plugins.platforms
# import cv2.qt.plugins.platforms.libqxcb
# cv2.qt.set_platformpluginpath("/usr/lib/x86_64-linux-gnu/qt6/plugins/platforms")


def name_in_dir(name):

    directories= os.listdir()


#    Print all the directory names
    for directory in directories:
        if(name==directory):
            return True
    return False
     
def add_image_to_file():
    return


#creates a capture object that ca[pptures videos
cap=cv2.VideoCapture(0)
#0 means that it will open the default camera in the system 

#if there isnt a camera
if not cap.isOpened():
    print("no camera")
    exit()

#if exists

while True:
    name=input("enter the name")
    if(not name_in_dir(name)):
        break
    
count=int(input("enter the number of images that have to be taken enter >0"))

while count>0:
    ret,frame=cap.read()
    #the image is captures ret which is a boolean value if true image is captured if false image hasnt been captured properly
    #the frame is  anumpy asrray fo all the pixel values  
    time.sleep(1)
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #conver the image to bgr format to be read by the imshow function
    image=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    cv2.imshow('image',frame)
    cv2.waitKey(1000)
    count=count-1

cap.release()
cv2.destroyAllWindows()
