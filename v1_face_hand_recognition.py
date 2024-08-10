import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh and Hands
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh()
hands = mp_hands.Hands()

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Open a video capture
cap = cv2.VideoCapture(0)  # 0 for the default webcam

class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def average_position(positions):
    x = sum([position.x for position in positions]) / len(positions)
    y = sum([position.y for position in positions]) / len(positions)
    return Position(x, y)

def distance_squared(p1, p2):
    return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for face and hand landmarks
    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    # Draw face landmarks
    lips = []
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            lip_positions = [
                face_landmarks.landmark[13],
                face_landmarks.landmark[14]
            ]

    # Draw hand landmarks
    fingertip_positions = []
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
            # Fingertip indices: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)
            fingertip_indices = [4, 8, 12, 16, 20]
            for fingertip_index in fingertip_indices:
                fingertip_positions.append(hand_landmarks.landmark[fingertip_index])

    if len(lip_positions) > 0 and len(fingertip_positions) > 0:
        lip_dist_squared = distance_squared(lip_positions[0], lip_positions[1])
        mouth_open = lip_dist_squared > 0.0001
        mouth_color = (0, 0, 255) if mouth_open else (255, 0, 0)
        cv2.line(frame,
            (int(lip_positions[0].x * frame.shape[1]), int(lip_positions[0].y * frame.shape[0])),
            (int(lip_positions[1].x * frame.shape[1]), int(lip_positions[1].y * frame.shape[0])),
            mouth_color, 2)

        mouth_position = average_position(lip_positions)
        fingertip_position = average_position(fingertip_positions)
        mouth_fingertip_dist_squared = distance_squared(mouth_position, fingertip_position)
        mouth_fingertip_close = mouth_fingertip_dist_squared < 0.01
        mouth_fingertip_color = (0, 0, 255) if mouth_fingertip_close else (0, 255, 0)
        cv2.line(frame,
            (int(mouth_position.x * frame.shape[1]), int(mouth_position.y * frame.shape[0])),
            (int(fingertip_position.x * frame.shape[1]), int(fingertip_position.y * frame.shape[0])),
            mouth_fingertip_color, 2)

        is_bite = mouth_open and mouth_fingertip_close
        cv2.putText(frame, f'Bite detected: {is_bite}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_bite else (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('MediaPipe Frame', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
