import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh and Hands
mp_face_detection = mp.solutions.face_detection

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def extract_face(image):
    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_detection.FaceDetection(min_detection_confidence=0.3) as face_detection:
        results = face_detection.process(rgb_image)

        if not results.detections:
            return

        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = image.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
        return image[y:y+h, x:x+w]
