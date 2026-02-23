# 🎭 Emotii în Timp Real — Real-Time Facial Emotion Recognition

A deep learning system that detects human facial emotions in real time using a custom-trained CNN built with TensorFlow/Keras.

**By: Iorga Mihai Liviu**

---

##  Overview

This project trains a Convolutional Neural Network (CNN) to classify facial expressions into 5 emotion categories. The model processes grayscale 48×48 pixel face images and outputs a probability distribution across emotion classes.
https://colab.research.google.com/drive/18j0Tet-kabSNBiJOHzxq_WsEz6jRcAz-

**Detected emotions:**
- 😠 Furie (Anger)
- 😨 Frică (Fear)
- 😊 Fericire (Happiness)
- 😢 Tristețe (Sadness)
- 😲 Surpriză (Surprise)

---

##  Dataset

- **Source:** [Human Face Emotions — Kaggle](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions)
- **Format:** Grayscale images, 48×48 pixels
- **Split:** 80% training / 20% validation

---

##  Model Architecture

A sequential CNN with the following structure:

| Layer | Details |
|---|---|
| Conv2D × 2 | 32 filters, 3×3, BatchNorm + ReLU |
| MaxPooling2D | 2×2 |
| Conv2D × 2 | 64 filters, 3×3, BatchNorm + ReLU |
| MaxPooling2D + Dropout(0.3) | |
| Conv2D | 128 filters, 3×3, BatchNorm + ReLU |
| MaxPooling2D | |
| Conv2D | 128 filters, 3×3, BatchNorm + ReLU |
| MaxPooling2D + Flatten | |
| Dense(512) | BatchNorm + ReLU + Dropout(0.5) |
| Dense(5) | Softmax output |

**Input shape:** `(48, 48, 1)`

---

##  Training

- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Max epochs:** 50 (with early stopping)
- **Callbacks:**
  - `EarlyStopping` — patience 5, restores best weights
  - `ReduceLROnPlateau` — patience 3, factor 0.1
- **Result:** Converged in ~22 epochs, reaching ~93%+ training accuracy

---

##  Usage

### 1. Install dependencies

```bash
pip install tensorflow pillow matplotlib numpy kagglehub
```

### 2. Download dataset & train

```bash
python expresion_recognition_v2.py
```

The script will:
- Download the dataset via `kagglehub`
- Train the model
- Save it as `mnist_cnn_model.h5`
- Plot loss and accuracy curves

### 3. Run inference on an image

Update the `image_path` variable in the script:

```python
image_path = "your_image.png"
```

The model will output the predicted emotion and a probability bar chart.

---

##  Results

The model achieves strong performance on validation data, with typical results:

-  High-confidence predictions (98–99%) on clear expressions
-  Lower confidence (~41%) on ambiguous or poorly-lit images
-  Accuracy drops to ~60% under poor lighting conditions
-  Occasional confusion between fear and surprise (visually similar expressions)

---

##  Feature Visualization

The project includes layer activation visualization — you can see how each convolutional layer processes the input:

- **Conv2D #1** — detects edges and basic contours
- **Conv2D #2** — captures textures
- **Conv2D #3+** — recognizes complex facial patterns
- **MaxPooling** — downsampled feature maps

---

##  Web Application (Flask)

The project includes a real-time web interface built with Flask and OpenCV that streams webcam video and overlays live emotion predictions.

### How it works

1. The webcam feed is captured with OpenCV
2. Haar Cascade detects faces in each frame
3. Detected faces are preprocessed (48×48 grayscale) and fed to the CNN
4. The predicted emotion and confidence are overlaid on the video stream
5. Faces are automatically saved to disk every ~25 frames (≈1 second at 30fps), organized by emotion

### Running the web app

```bash
pip install flask opencv-python tensorflow numpy
python app.py
```

Then open your browser at: [http://localhost:5000](http://localhost:5000)

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main page with live video feed |
| `/video_feed` | GET | MJPEG video stream |
| `/emotion` | GET | Current emotion as JSON `{"emotion": "happy", "confidence": 97.3}` |
| `/save_current` | GET | Manually trigger saving the current face |

### Saved Images

Detected faces are saved automatically to the `saved_faces/` folder, organized by emotion:

```
saved_faces/
├── happy/
│   └── happy_97.3_20240223_143012_123456.jpg
├── sad/
│   └── sad_84.1_20240223_143025_654321.jpg
└── ...
```

Each filename encodes the emotion, confidence score, and timestamp.

---

##  Project Structure

```
├── expresion_recognition_v2.py   # Model training + inference + visualization
├── app.py                        # Flask web application
├── templates/
│   └── index.html                # Web interface template
├── saved_faces/                  # Auto-saved detected faces (generated at runtime)
├── mnist_cnn_model.h5            # Saved model (generated after training)
└── README.md
```

---

##  Future Work

- Add more emotion classes (contempt, confusion, disgust)
- Multi-face detection in a single frame
- Real-time webcam integration
- Potential applications: online education, mental health monitoring, gaming & VR

---

## 📄 License

This project was created for educational purposes.
