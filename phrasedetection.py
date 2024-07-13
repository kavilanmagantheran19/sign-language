import cv2
import numpy as np
from keras.models import model_from_json
import time
from transformers import AlbertTokenizer

# Initialize the tokenizer
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

# Load the sign language model
json_file = open("sign_language_detection_model2.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("sign_language_model2.keras")

# Extract features
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 128, 128, 1)
    return feature / 255.0

# Initialize video capture and labels
cap = cv2.VideoCapture(0)
label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','blank']
sequence = ""  # To store the sequence of recognized letters
last_prediction = ""
same_count = 0
max_same_count = 5  # Number of frames to wait before adding the same letter to the sequence

# Delay configuration
delay_seconds = 3  # 3 seconds delay
last_update_time = time.time()

# Function to update sequence based on accuracy with delay
def update_sequence_with_delay(prediction_label, accu):
    global sequence, last_prediction, same_count, last_update_time

    current_time = time.time()

    if accu >= 90:
        if prediction_label == last_prediction:
            same_count += 1
            if same_count >= max_same_count and (current_time - last_update_time) >= delay_seconds:
                sequence += prediction_label
                same_count = 0
                last_update_time = current_time
        else:
            same_count = 0
        last_prediction = prediction_label

# Main loop
while True:
    _, frame = cap.read()
    cv2.rectangle(frame, (0, 40), (500, 500), (255, 255, 255), 2)
    cropframe = frame[40:500, 0:500]
    cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
    cropframe = cv2.resize(cropframe, (128, 128))
    cropframe = extract_features(cropframe)
    pred = model.predict(cropframe)
    prediction_label = label[pred.argmax()]
    accu = np.max(pred) * 100  # Calculate the accuracy

    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)

    if prediction_label == 'blank':
        cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        accu_str = "{:.2f}".format(accu)
        cv2.putText(frame, f'{prediction_label}  {accu_str}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Update sequence based on accuracy with delay
        update_sequence_with_delay(prediction_label, accu)

    # Display the sequence of letters
    output_frame = np.zeros((frame.shape[0] + 100, frame.shape[1], 3), dtype=np.uint8)
    output_frame[:frame.shape[0], :frame.shape[1]] = frame

    cv2.putText(output_frame, f'Sequence: {sequence}', (10, frame.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Tokenize the detected phrase
    if sequence:
        tokens = tokenizer.tokenize(sequence)
        token_str = " ".join(tokens)
        cv2.putText(output_frame, f'Tokens: {token_str}', (10, frame.shape[0] + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Sign Language Detection", output_frame)
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
