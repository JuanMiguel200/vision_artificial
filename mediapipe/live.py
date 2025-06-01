import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
'''
hands = mp.solutions.hands.Hands(
    max_num_hands=2,  # Aquí defines el número de manos
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
            )

    cv2.imshow('Manos detectadas', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
'''
# Configurar opciones del modelo
model_path = 'model/gesture_recognizer.task'
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Función de callback para procesar resultados
def gesture_recognizer_callback(result, output_image, timestamp_ms):
    # Convertir imagen de MediaPipe a OpenCV
    annotated_frame = np.copy(output_image.numpy_view())

    # Dibujar landmarks
    if result.hand_landmarks:
        for landmarks in result.hand_landmarks:
            for lm in landmarks:
                x = int(lm.x * annotated_frame.shape[1])
                y = int(lm.y * annotated_frame.shape[0])
                cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)

    # Dibujar nombre del gesto
    if result.gestures and result.gestures[0]:
        top_gesture = result.gestures[0][0]
        gesture_name = top_gesture.category_name
        score = top_gesture.score
        text = f'{gesture_name} ({score:.2f})'
        cv2.putText(annotated_frame, text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    # Mostrar imagen con anotaciones
    cv2.imshow("Reconocimiento de Gestos", annotated_frame)
    cv2.waitKey(1)

# Configurar GestureRecognizer en modo STREAM
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
   
    result_callback=gesture_recognizer_callback
)

# Crear reconocedor
with GestureRecognizer.create_from_options(options) as recognizer:
    cap = cv2.VideoCapture(0)
    timestamp = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("No se pudo leer la cámara.")
            break

        # Convertir a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Crear objeto Image de MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Enviar imagen al reconocedor
        recognizer.recognize_async(mp_image, timestamp)
        timestamp += 33  # Aprox. 30 FPS → 33ms entre frames

    cap.release()
    cv2.destroyAllWindows()
