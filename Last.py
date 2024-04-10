from LastFuncCalMain import *
import mimetypes
from tkinter import *  
from tkinter import messagebox  
from tkinter import filedialog

mimetypes.init()
root=Tk()
variable1=StringVar()    
variable2=StringVar()    

root.geometry("800x800")

l1 =Label(root, text = "Biomechanical Posture", font= ('Helvetica 25 bold')).place(relx=.5, rely=0,anchor= N)
l2 =Label(root, textvariable = variable1, font= ('Helvetica 10 bold')).place(relx=.5, rely=.6,anchor= N)
l3 =Label(root, textvariable = variable2, font= ('Helvetica 10 bold')).place(relx=.5, rely=.7,anchor= N)
b1=Button(root,text="Browse for a video or an audio",font=40,command=video_pose_estimation2(0)
).place(relx=.5, rely=.2,anchor= N)
#b1=Button(root,text="Choose Live Posture Analysis using webcam",font=40,command=webcam).place(relx=.5, rely=.4,anchor= N)
root.mainloop()
#video_pose_estimation2('vid1.mp4')
