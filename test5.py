import customtkinter
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from tkVideoPlayer import TkinterVideo
customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

app = customtkinter.CTk()  # create CTk window like you do with the Tk window
app.geometry("1280*720")

# def button_function():
#     print("button pressed")

# Use CTkButton instead of tkinter Button


leftframe=customtkinter.CTkFrame(master=app ,width=300,height=720,border_color="yellow",border_width=5)
leftframe.pack(side="left",expand=False,fill="y")

rightframe=customtkinter.CTkFrame(master=app ,border_color="orange",border_width=10)
rightframe.pack(side="left",expand=True,fill="both")

subframe=customtkinter.CTkFrame(master=rightframe,width=500,height=540,border_color="red",border_width=2)
subframe.pack(side="top",expand=True,fill="both")

subfram2=customtkinter.CTkFrame(master=rightframe,width=500,height=180,border_color="blue",border_width=15)
subfram2.pack(side="bottom",expand=False,fill="both")


#############################################################################################33
def open_video():
    # for widget in frame_1.winfo_children():
    #     widget.destroy()
    vid_player.stop()
    global video_file
    video_file=filedialog.askopenfilename(filetypes =[('Video', ['*.mp4','*.avi','*.mov','*.mkv','*gif']),('All Files', '*.*')])
    if video_file:
        try:
            vid_player.load(video_file)
            vid_player.play()
            progress_slider.set(-1)
            play_pause_btn.configure(text="Pause ||")
        except:
            print("Unable to load the file")

def update_duration(event):
    try:
        duration = int(vid_player.video_info()["duration"])
        progress_slider.configure(from_=-1, to=duration, number_of_steps=duration)
    except:
        pass
    
def seek(value):
    if video_file:
        try:
            
            vid_player.seek(int(value))
            vid_player.play()
            vid_player.after(50,vid_player.pause)
            play_pause_btn.configure(text="Play ►")
        except:
            pass
    
def update_scale(event):
    try:
        progress_slider.set(int(vid_player.current_duration()))
    except:
        pass
    
def play_pause():
    if video_file:
        if vid_player.is_paused():
            vid_player.play()
            play_pause_btn.configure(text="Pause ||")

        else:
            vid_player.pause()
            play_pause_btn.configure(text="Play ►")
        
def video_ended(event):
    play_pause_btn.configure(text="Play ►")
    progress_slider.set(-1)



video_file=''
frame_1 = customtkinter.CTkFrame(master=subframe, corner_radius=15,border_color="green")
frame_1.pack(pady=20, padx=20, fill="both", expand=True)

button_1 = customtkinter.CTkButton(master=leftframe, text="Open Video", corner_radius=8, command=open_video)
button_1.place(relx=0.5, rely=0.3, anchor="n") 

vid_player = TkinterVideo(master=frame_1, scaled=True, keep_aspect=True, consistant_frame_rate=True, bg="black",)
vid_player.set_resampling_method(1)
vid_player.pack(expand=True, fill="both", padx=10, pady=10)
vid_player.bind("<<Duration>>", update_duration)
vid_player.bind("<<SecondChanged>>", update_scale)
vid_player.bind("<<Ended>>", video_ended)

progress_slider = customtkinter.CTkSlider(master=frame_1, from_=-1, to=1, number_of_steps=1, command=seek)
progress_slider.set(-1)
progress_slider.pack(fill="both", padx=10, pady=10)

play_pause_btn = customtkinter.CTkButton(master=frame_1, text="Play ►", command=play_pause)
play_pause_btn.pack(pady=10)
    

######################################################################################################3


def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif;*.bmp")])
    if file_path:
        resize_image(file_path)

# Function to resize and display the image
def resize_image(file_path):
    
    img = Image.open(file_path)
    img = img.resize((800, 600), Image.LANCZOS)
    ctk_image = customtkinter.CTkImage(light_image=img, dark_image=img, size=(500, 540))
    
    # Clear previous content in subframe
    # for widget in subframe.winfo_children():
    #     widget.destroy()
    
    # Create a label to display the image
    label = customtkinter.CTkLabel(master=frame_1, image=ctk_image, text="")
      # Keep a reference to avoid garbage collection
    label.place(relx=0.5, rely=0.5, anchor="center")


# Function to capture video from webcam and display it in real-time
def display_video():
    
    for widget in subframe.winfo_children():
        widget.destroy()
    cap = cv2.VideoCapture(0)  # Index 0 represents the default webcam

    while True:
        ret, frame = cap.read()  # Read a frame from the webcam

        if not ret:
            break

        # Convert the frame from BGR to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the frame to a PIL image
        pil_image = Image.fromarray(frame_rgb)
        
        # Resize the image to fit the subframe dimensions
        resized_image = pil_image.resize((800, 600), Image.LANCZOS)
        
        # Convert the PIL image to a format suitable for customtkinter
        ctk_image = ImageTk.PhotoImage(resized_image)
        
        # Create a label to display the image in the subframe
        if not hasattr(display_video, 'label'):  # Check if label has been created
            display_video.label = customtkinter.CTkLabel(master=frame_1, image=ctk_image, text="")
            display_video.label.image = ctk_image  # Keep a reference to avoid garbage collection
            display_video.label.place(relx=0.5, rely=0.5, anchor="center")
        else:
            display_video.label.configure(image=ctk_image)  # Update the image in the existing label
        
        # Update the main loop to refresh the display
        app.update_idletasks()
        app.update()

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows when the loop exits



b1 = customtkinter.CTkButton(master = leftframe, text = "Webcam",command=display_video)
b1.place(relx=0.5, rely=0.1, anchor="n") 
# b1.pack(side="top",padx=10,pady=10) 
b2 = customtkinter.CTkButton(master = leftframe, text = "Browse", command=browse_image)
b2.place(relx=0.5, rely=0.2, anchor="n") 
# button = customtkinter.CTkButton(master=leftframe, text="CTkButton", command=button_function)
# button.place(relx=0.5, rely=0.5, anchor=customtkinter.CENTER)
app.mainloop()





