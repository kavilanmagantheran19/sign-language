from keras.models import model_from_json
import cv2
import numpy as np
import tkinter as tk
from tkinter import Text




json_file = open("sign_language_detection_model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("sign_language_model.keras")

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,128,128,1)
    return feature/255.0

def update_textarea(textarea, gesture, accuracy):
    textarea.insert(tk.END, f'{gesture} ({accuracy}%)\n')
    textarea.see(tk.END)

#GUI CREATION
root = tk.Tk()
root.title("Sign Language Recognition")

# Create a frame for the video and text area
frame = tk.Frame(root)
frame.pack()

# Create a canvas for the video feed
canvas = tk.Canvas(frame, width=640, height=480)
canvas.pack()

# Create a text area for recognized gestures
text_area = Text(frame, height=10, width=50)
text_area.pack()

cap = cv2.VideoCapture(0)
label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
while True:
    _,frame = cap.read()
    cv2.rectangle(frame,(0,40),(500,500),(255,255,255), 2)
    cropframe = frame[40:500, 0:500]
    cropframe = cv2.cvtColor(cropframe,cv2.COLOR_BGR2GRAY)
    cropframe = cv2.resize(cropframe,(128,128))
    cropframe = extract_features(cropframe)
    pred = model.predict(cropframe)
    prediction_label = label[pred.argmax()]
    cv2.rectangle(frame, (0,0), (300, 40), (0, 165, 255), -1)
    if prediction_label == 'blank':
        cv2.putText(frame, " ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255),2,cv2.LINE_AA)
    else:
        accu = "{:.2f}".format(np.max(pred)*100)
        cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255),2,cv2.LINE_AA)
    cv2.imshow("output",frame)
    # cv2.waitKey(27)

    # Check if the "Escape" key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()