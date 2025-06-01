import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

# Cargar imagen con OpenCV
image_path = 'image/victory.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convertir a ImageFormat de MediaPipe
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

#modelo y variables base de mediapipe
model_path = '/model/gesture_recognizer.task'
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the image mode:
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='model/gesture_recognizer.task'),
    running_mode=VisionRunningMode.IMAGE)
with GestureRecognizer.create_from_options(options) as recognizer:
    result = recognizer.recognize(mp_image)

    # Mostrar resultados de consola
    print(result)

    # Dibujar resultados en la imagen
    annotated_image = image.copy()

    # Dibujar landmarks si existen
    if result.hand_landmarks:
        for landmarks in result.hand_landmarks:
            for landmark in landmarks:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

    # Escribir nombre del gesto más confiable
    if result.gestures and result.gestures[0]:
        top_gesture = result.gestures[0][0]
        gesture_name = top_gesture.category_name
        score = top_gesture.score
        text = f'Gesto: {gesture_name} ({score:.2f})'
        cv2.putText(annotated_image, text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Mostrar imagen con anotaciones
    cv2.imshow('Gesto Detectado', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()