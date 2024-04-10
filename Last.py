from LastFuncCalMain import *
import mimetypes
from tkinter import *  
from tkinter import filedialog
import customtkinter

root = customtkinter.CTk()

root.geometry ("300x400")

# def submit():
#     # Get the input value from the entry widget
#     input_text = entry.get()
#     # Do something with the input value, such as printing it
#     print("Input:", input_text)

def video_pose_estimation():
    video_pose_estimation2(0)

def browse():
    filename = filedialog.askopenfilename()
    mimestart = mimetypes.guess_type(str(filename))[0]

    if mimestart != None:
        mimestart = mimestart.split('/')[0]
    if mimestart == 'video' or mimestart == 'image':
        video_pose_estimation2(str(filename))
    else:
        pass

b1 = customtkinter.CTkButton(master = root, text = "Webcam", command = video_pose_estimation).place(relx=0.5, rely=.2, anchor = CENTER)
b2 = customtkinter.CTkButton(master = root, text = "Browse", command = browse).place(relx=0.5, rely=.4, anchor = CENTER)
# b3 = customtkinter.CTkButton(master = root, text = "Browse", command = webcam).place(relx=0.5, rely=.2, anchor = CENTER)

root. mainloop()






mimetypes.init()

# video_pose_estimation2(0)
