import math as m
import cv2
import numpy as np
import time
import mediapipe as mp
import matplotlib.pyplot as plt
from MainTK import *
import time
import os
import mimetypes
from tkinter import *  
from tkinter import messagebox  
from tkinter import filedialog
import pyautogui as pgi





mimetypes.init()
root=Tk()
variable1=StringVar()    
variable2=StringVar()    

root.geometry("800x800")

l1 =Label(root, text = "Biomechanical Posture", font= ('Helvetica 25 bold')).place(relx=.5, rely=0,anchor= N)
l2 =Label(root, textvariable = variable1, font= ('Helvetica 10 bold')).place(relx=.5, rely=.6,anchor= N)
l3 =Label(root, textvariable = variable2, font= ('Helvetica 10 bold')).place(relx=.5, rely=.7,anchor= N)

# pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, model_complexity=1)
pose_video = mp_pose.Holistic(static_image_mode=False, min_detection_confidence=0., model_complexity=1)

def sendWarning(x):
    pass

# =============================CONSTANTS and INITIALIZATIONS=====================================#
# Initilize frame counters.
good_frames = 0
bad_frames = 0
# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX
# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
yellow = (0, 255, 255)
pink = (255, 0, 255)
white = (255,255,255)
# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# ===============================================================================================#


def video_pose_estimation(name):
    camera_video = cv2.VideoCapture(name)
    # Iterate until the webcam is accessed successfully.
    while camera_video.isOpened():

        # Read a frame.
        ok, frame = camera_video.read()
        
        # Check if frame is not read properly.
        if not ok:
            
            # Continue to the next iteration to read the next frame and ignore the empty camera frame.
            continue
        
        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)
        # Get the width and height of the frame
        frame_height, frame_width, _ =  frame.shape
        # Resize the frame while keeping the aspect ratio.
        frame = cv2.resize(frame, (int(frame_width * (960 / frame_height)), 960))
        # print(frame.shape)
        t1 = time.time()
        # Perform Pose landmark detection.
        frame, landmarks = detectPose(frame, pose_video, display=False)
        # Get fps.
        fps = camera_video.get(cv2.CAP_PROP_FPS)
        # Get height and width.
        frame_height, frame_width = frame.shape[:2]
        # Convert the BGR frame to RGB.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame.
        keypoints = pose.process(frame)
        # Convert the frame back to BGR.
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Use lm and lmPose as representative of the following methods.

        # Check if the landmarks are detected.
        if landmarks:
            
            # Perform the Pose Classification.
            frame, _ = classifyPose(landmarks, frame, display=False)

        t2 = time.time() - t1
        cv2.putText(frame, "{:.0f} ms".format(
                t2*1000), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (203, 52, 247), 1)

        cv2.imshow('Pose Classification', frame)
        Weight = 10
        LC, RC = find_rula_opp('TableA.csv','TableB.csv','TableC.csv')
        variable1.set("Left RULA grand score: " + str(LC))
        variable2.set("Right RULA grand score: " + str(RC))
        root.update()

        # Wait until a key is pressed.
        # Retreive the ASCII code of the key pressed
        k = cv2.waitKey(1) & 0xFF
        # Check if 'ESC' is pressed.
        if k == 27 and landmarks:
        # Break the loop.
            break
        
    # Release the VideoCapture object and close the windows.
    camera_video.release()
    cv2.destroyAllWindows()

def image_pose_estimation(file_path):
    frame = cv2.imread(file_path)

    frame_height, frame_width, _ = frame.shape

    #frame = cv2.resize(frame, (int(frame_width * (960 / frame_height)), 960))

    t1 = time.time()
    frame, landmarks = detectPose(frame, pose_video, display=False)
    fps = 1 / (time.time() - t1)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    keypoints = pose.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if landmarks:
        frame, _ = classifyPose(landmarks, frame, display=False)

    cv2.putText(frame, "{:.0f} fps".format(fps), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (203, 52, 247), 1)
    cv2.imshow('Pose Classification', frame)

    LC, RC = find_rula_opp('TableA.csv','TableB.csv','TableC.csv')
    variable1.set("Left RULA grand score: " + str(LC))
    variable2.set("Right RULA grand score: " + str(RC))
    root.update()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def webcam():
    video_pose_estimation(0)

def browsefunc():
   
   filename =filedialog.askopenfilename()
   mimestart = mimetypes.guess_type(str(filename))[0]

   if mimestart != None:
      mimestart = mimestart.split('/')[0]

   if mimestart == 'video':
      video_pose_estimation(str(filename))
   elif mimestart == 'image':
      image_pose_estimation(str(filename))
   else:
      pass
   
b1=Button(root,text="Browse for a video or an audio",font=40,command=browsefunc).place(relx=.5, rely=.2,anchor= N)
b1=Button(root,text="Choose Live Posture Analysis using webcam",font=40,command=webcam).place(relx=.5, rely=.4,anchor= N)
root.mainloop()