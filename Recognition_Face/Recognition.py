from tkinter import *
import cv2

face_cascade_db = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
root = Tk()

text = StringVar()

def open_video():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade_db.detectMultiScale(img_gray, 1.1, 17)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (12, 22, 0), 2)
        cv2.imshow('Result Video', img)
        if cv2.waitKey(1) == 27:
            break
    cap = release()
    cv2.destroyAllWindows()

def open_photo():
    img = cv2.imread(f"../Recognition_Face/Faces/{text.get()}.png") 
    faces = face_cascade_db.detectMultiScale(img, 1.1, 17)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (12, 22, 0), 2)
    cv2.imshow(f'Result Photo {text.get()}.png', img)
    cv2.imwrite(f"Recognition_save/{text.get()}.png", img)

# --- 
root.title("Recognition faces")
root.geometry("300x130")
root.resizable(width=False, height=False) 

# Text 
info = Label(text="Enter the name of the file \n located in the folder 'Faces'")
info.pack()

# Entry field
text_entry = Entry(textvariable=text)
text_entry.pack()

# 1 buttom
button1 = Button(text="Display selected photo", command=open_photo)
button1.pack()

# 2 buttom
button2 = Button(text="Face recognition from the camera", command=open_video)
button2.pack()

# Text
info = Label(text="To turn on the video press 'ESC'", fg="red")
info.pack()

# --- Output ---
root.mainloop()


