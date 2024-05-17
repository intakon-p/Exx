import customtkinter
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
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
    for widget in subframe.winfo_children():
        widget.destroy()
    
    # Create a label to display the image
    label = customtkinter.CTkLabel(master=subframe, image=ctk_image, text="")
      # Keep a reference to avoid garbage collection
    label.pack(expand=True, fill="both")


# Function to capture video from webcam and display it in real-time
def display_video():
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
        resized_image = pil_image.resize((subframe.winfo_width(), subframe.winfo_height()), Image.ANTIALIAS)
        
        # Convert the PIL image to a format suitable for customtkinter
        ctk_image = ImageTk.PhotoImage(resized_image)
        
        # Create a label to display the image in the subframe
        if not hasattr(display_video, 'label'):  # Check if label has been created
            display_video.label = customtkinter.CTkLabel(master=subframe, image=ctk_image, text="")
            display_video.label.image = ctk_image  # Keep a reference to avoid garbage collection
            display_video.label.pack(expand=True, fill="both")
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