# 🤚 Hand Gesture Recognition Using Deep Learning

A real-time hand gesture recognition system developed using Convolutional Neural Networks (CNN), TensorFlow, Keras, and OpenCV to recognize American Sign Language (ASL) alphabet gestures through a webcam.

## 📌 Project Overview

Hand gestures are an important form of non-verbal communication. However, communication can become challenging when people are unfamiliar with sign language.

This project aims to bridge this communication gap by developing a deep learning-based hand gesture recognition system.

A CNN model is trained to classify hand gesture images into **29 different classes**, including the 26 English alphabet letters (A–Z) and three additional classes: Space, Delete, and Nothing.

The trained model is integrated with OpenCV to perform real-time gesture recognition using a webcam.

## 🎯 Objectives

- Develop a CNN-based hand gesture classification model.
- Recognize 29 different ASL gesture classes.
- Perform image preprocessing for accurate gesture classification.
- Implement real-time hand gesture detection using a webcam.
- Convert visual hand gestures into their corresponding predicted labels.

## ✨ Key Features

- **Deep Learning Classification:** Uses a CNN model to identify hand gestures.
- **29 Gesture Classes:** Supports A–Z, Space, Delete, and Nothing.
- **Real-Time Detection:** Uses a webcam to capture and recognize gestures.
- **Computer Vision:** Uses OpenCV for image acquisition and processing.
- **Trained Model:** Saves the trained model in Keras format for inference.

## 🛠️ Technologies Used

| Technology | Application |
|---|---|
| Python | Core programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN model development |
| OpenCV | Image processing and webcam access |
| NumPy | Numerical computations |
| Matplotlib | Training visualization |
| Jupyter Notebook | Model training and experimentation |

## 🧠 Model Description

The project uses a **Convolutional Neural Network (CNN)** for hand gesture image classification.

CNNs are effective for image recognition because they automatically learn important visual features such as edges, shapes, textures, and patterns.

The model learns the visual characteristics of different hand gestures during training and uses these learned features to predict the class of an input gesture.

### Gesture Classes

The model supports 29 output classes:

- English alphabets: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z.
- Additional classes: Space, Delete, Nothing.

## ⚙️ System Workflow

```text
       Webcam Input
             |
             v
      Capture Image
             |
             v
      Image Processing
             |
             v
       Preprocessing
             |
             v
       Trained CNN Model
             |
             v
      Gesture Prediction
             |
             v
      Display Predicted
          Gesture
```

## 📂 Project Structure

```text
hand_gesture_recognition/
│
├── README.md
│
├── cnn_train_model01.ipynb
│
├── hand_gesture_29class.keras
│
└── real_time_detection.py
```

### File Description

| File | Description |
|---|---|
| `cnn_train_model01.ipynb` | Notebook containing CNN model training and experimentation |
| `hand_gesture_29class.keras` | Saved trained CNN model for 29-class classification |
| `real_time_detection.py` | Python script for real-time webcam-based gesture recognition |
| `README.md` | Project documentation |

## 🚀 Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/lovelypriya2005/hand_gesture_recognition.git
```

### 2. Navigate to the Project Directory

```bash
cd hand_gesture_recognition
```

### 3. Install Required Libraries

```bash
pip install tensorflow opencv-python numpy matplotlib
```

### 4. Run the Real-Time Detection Script

```bash
python real_time_detection.py
```

Make sure your webcam is connected and the trained model path in the Python script points to the correct model file.

## 📊 Results

The trained CNN model performs classification across 29 hand gesture categories.

The real-time detection script uses the saved model to predict gestures from webcam input.

*Model evaluation metrics and sample prediction results can be added here after final testing.*

## 🔮 Future Enhancements

- Extend recognition from individual alphabets to complete words and sentences.
- Improve recognition under different lighting and background conditions.
- Develop a graphical user interface for easier interaction.
- Integrate text-to-speech conversion.
- Improve prediction speed and robustness.
- Explore continuous sign language recognition.

## 👩‍💻 Author

**Lovely Priya**

GitHub: [@lovelypriya2005](https://github.com/lovelypriya2005)

---

⭐ If you find this project interesting, consider giving the repository a star!
