from LastFuncCalMain import *
import mimetypes
from tkinter import *  
import customtkinter

root = customtkinter.CTk()

root.geometry ("300x400")

def webcam():
    video_pose_estimation2(0)

b1 = customtkinter.CTkButton(master = root, text = "Browse", command = webcam())

b1.place(relx=0.5, rely=.5, anchor = CENTER)

root. mainloop()






mimetypes.init()

# video_pose_estimation2(0)
