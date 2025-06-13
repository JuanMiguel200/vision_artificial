import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
import queue

# ---- SETUP ----
model_path = '/model/gesture_recognizer.task' #ubicacion del modelo
BaseOptions = mp.tasks.BaseOptions #importa configuraciones por defecto
GestureRecognizer = mp.tasks.vision.GestureRecognizer #tipo de tarea
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions #objeto que almacena las opciones
VisionRunningMode = mp.tasks.vision.RunningMode #modo de obtención de los frames (imagen, video, en vivo)

result_queue = queue.Queue()
 # marca los 21 puntos clave de la mano y escribe el gesto reconocido recibe en result un array con las coordenadas y el gesto detectado y en output_image el frame en el que esto ocurrió
def gesture_recognizer_callback(result, output_image, timestamp_ms):
    annotated_frame = np.copy(output_image.numpy_view())

    if result.hand_landmarks:
        for landmarks in result.hand_landmarks:
            for lm in landmarks:
                x = int(lm.x * annotated_frame.shape[1])
                y = int(lm.y * annotated_frame.shape[0])
                cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)

    if result.gestures and result.gestures[0]:
        top_gesture = result.gestures[0][0]
        gesture_name = top_gesture.category_name
        score = top_gesture.score
        text = f'{gesture_name} ({score:.2f})'
        cv2.putText(annotated_frame, text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    # Enviar imagen anotada al hilo principal
    result_queue.put(annotated_frame)

# ---- MAIN LOOP ----
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=gesture_recognizer_callback
) #se definen las configuraciones base del reconocimiento

with GestureRecognizer.create_from_options(options) as recognizer:
    cap = cv2.VideoCapture(0)
    timestamp = 0
    #Llama a opencv para capturar las imagenes
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("No se pudo leer la cámara.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#convierte la imagen de bgr a rgb para ser procesada por mediapipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Enviar imagen
        recognizer.recognize_async(mp_image, timestamp)
        timestamp += 33

        # Mostrar frame anotado si está disponible
        if not result_queue.empty():
            annotated = result_queue.get()
            bgr_frame = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            cv2.imshow("Reconocimiento de Gestos", bgr_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

