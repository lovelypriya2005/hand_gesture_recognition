
import cv2
import numpy as np
import tensorflow as tf
import os

# Find the model in the same folder as this Python script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_gesture_29class.keras")

# Load the trained 29-class model
model = tf.keras.models.load_model(MODEL_PATH)

labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

# Verify model and label count
if model.output_shape[-1] != len(labels):
    raise ValueError(
        f"Model has {model.output_shape[-1]} outputs, "
        f"but there are {len(labels)} labels."
    )

print("Model loaded successfully!")
print("Number of gesture classes:", len(labels))
print("Press Q to quit the webcam.")

# Open the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open the webcam. Check camera permissions "
        "and close other apps using the camera."
    )

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not capture a webcam frame.")
            break

        # Mirror the image for a more natural webcam experience
        frame = cv2.flip(frame, 1)

        # Region of interest (ROI)
        x1, y1, x2, y2 = 100, 100, 300, 300
        roi = frame[y1:y2, x1:x2]
        cv2.imshow("Hand Crop - Model Input", roi)

        if roi.size == 0:
            continue

        # Match the model's training input: 50 x 50 RGB-channel image
        img = cv2.resize(roi, (50, 50))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict the gesture
        
        prediction = model.predict(img, verbose=0)[0]

        # Get the three most likely classes
        top_indices = np.argsort(prediction)[-3:][::-1]

        char_index = int(top_indices[0])
        predicted_label = labels[char_index]
        confidence = float(prediction[char_index]) * 100

       # Print top three predictions in the terminal
        top_three = [
             f"{labels[i]}: {prediction[i] * 100:.1f}%"
            for i in top_indices
        ]
        print(" | ".join(top_three))


        # Display the prediction
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        display_label = predicted_label
        if predicted_label == 'space':
            display_label = 'SPACE'
        elif predicted_label == 'del':
            display_label = 'DELETE'
        elif predicted_label == 'nothing':
            display_label = 'NOTHING'

        text = f"{display_label}: {confidence:.1f}%"
        cv2.putText(
            frame, text, (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 255, 255), 2
        )

        cv2.imshow("Hand Gesture Recognition - 29 Classes", frame)

        # Press Q to close
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
