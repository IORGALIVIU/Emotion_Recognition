from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf
import os
from datetime import datetime

app = Flask(__name__)

# Creează folderul pentru salvarea imaginilor
SAVE_FOLDER = 'saved_faces'
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Încarcă modelul antrenat
model = keras.models.load_model('mnist_cnn_model.h5')

# Etichetele emoțiilor (în ordinea din datasetul tău)
EMOTIONS = ['angry', 'fear', 'happy', 'sad', 'surprise']

# Încarcă clasificatorul Haar Cascade pentru detectarea feței
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Variabilă globală pentru emoția detectată
current_emotion = "Nicio față detectată"
confidence = 0.0

# Contor pentru salvarea imaginilor
frame_counter = 0
SAVE_INTERVAL = 25  # Salvează o imagine la fiecare 30 de frame-uri (aproximativ 1 secundă la 30fps)


def save_face_image(face_img, emotion, confidence_val):
    """
    Salvează imaginea feței preprocesate în folder
    """
    try:
        # Creează subfolder pentru emoție
        emotion_folder = os.path.join(SAVE_FOLDER, emotion)
        if not os.path.exists(emotion_folder):
            os.makedirs(emotion_folder)

        # Generează nume de fișier cu timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{emotion}_{confidence_val:.1f}_{timestamp}.jpg"
        filepath = os.path.join(emotion_folder, filename)

        # Salvează imaginea
        cv2.imwrite(filepath, face_img)
        print(f"Imagine salvată: {filepath}")

    except Exception as e:
        print(f"Eroare la salvarea imaginii: {e}")


def preprocess_face(face_img):
    """
    Preprocesează imaginea feței pentru model
    """
    # Redimensionează la 48x48
    face_resized = cv2.resize(face_img, (48, 48))

    # Convertește la grayscale dacă nu este deja
    if len(face_resized.shape) == 3:
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_resized

    # Normalizează (0-255 -> 0-1)
    face_normalized = face_gray

    # Adaugă dimensiunile necesare: (1, 48, 48, 1)
    face_input = np.expand_dims(face_normalized, axis=-1)
    face_input = np.expand_dims(face_input, axis=0)

    return face_input, face_resized  # Returnează și imaginea redimensionată pentru salvare


def generate_frames():
    """
    Generează frame-uri video cu detecție de emoții
    """
    global current_emotion, confidence, frame_counter

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Eroare: Nu se poate accesa camera")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_counter += 1

        # Convertește frame-ul la grayscale pentru detecția feței
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectează fețele
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48)
        )

        if len(faces) > 0:
            # Folosește prima față detectată
            (x, y, w, h) = faces[0]

            # Extrage regiunea feței
            face_roi = gray[y:y + h, x:x + w]

            # Preprocesează fața
            face_input, face_resized = preprocess_face(face_roi)

            # Prezice emoția
            predictions = model.predict(face_input, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][emotion_idx]) * 100
            current_emotion = EMOTIONS[emotion_idx]

            # Salvează imaginea la interval regulat
            if frame_counter % SAVE_INTERVAL == 0:
                save_face_image(face_resized, current_emotion, confidence)

            # Desenează dreptunghi în jurul feței
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Afișează emoția și confidența
            label = f"{current_emotion}: {confidence:.1f}%"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            current_emotion = "Nicio față detectată"
            confidence = 0.0

        # Encodează frame-ul ca JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame-ul în format multipart
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()


@app.route('/')
def index():
    """
    Pagina principală
    """
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """
    Route pentru streaming video
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/emotion')
def get_emotion():
    """
    Route pentru a obține emoția curentă (JSON)
    """
    return jsonify({
        'emotion': current_emotion,
        'confidence': round(confidence, 2)
    })


@app.route('/save_current')
def save_current():
    """
    Route pentru a salva manual imaginea curentă
    """
    global current_emotion, confidence
    if current_emotion != "Nicio față detectată":
        return jsonify({
            'success': True,
            'message': 'Imaginea va fi salvată la următorul frame'
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Nu există față detectată'
        })


if __name__ == '__main__':
    print("Pornire aplicație Flask...")
    print(f"Imaginile vor fi salvate în folderul: {SAVE_FOLDER}")
    print("Accesează aplicația la: http://localhost:5000")
    app.run(debug=True, threaded=True)
