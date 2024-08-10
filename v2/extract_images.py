from collections import deque
import cv2
from extract_face import extract_face
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('training3.h5')
input_size = (224, 224)

# Open a video capture
cap = cv2.VideoCapture(0)  # 0 for the default webcam
# cap = cv2.VideoCapture('raw_session.mp4')

i = -1

bite_probability_deque = deque(maxlen=3)

while cap.isOpened():
    ret, image = cap.read()
    i += 1
    if not ret:
        break
    # Flip the image horizontally
    image = cv2.flip(image, 1)

    extracted_face = extract_face(image)
    if extracted_face is None or extracted_face.shape[0] == 0 or extracted_face.shape[1] == 0:
        continue

    # cv2.imwrite(f'frames/{i}.jpg', extracted_face)
    # cv2.imshow('Image', extracted_face)
    # print('\n\n\n', i, '\n\n\n')

    resized_image = cv2.resize(extracted_face, input_size)
    normalized_image = resized_image / 255.0

        # Add a batch dimension since the model expects batches of images
    input_image = np.expand_dims(normalized_image, axis=0)

    # Run the model on the input image
    prediction = model.predict(input_image)
    bite_probability = 1 - prediction[0][0]
    bite_probability_deque.append(bite_probability)
    smoothed_bite_probability = np.mean(bite_probability_deque)
    is_bite = smoothed_bite_probability > 0.7

    cv2.putText(image, f'Current bite probability: {round(bite_probability * 100, 1)}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if is_bite else (255, 0, 0), 2)
    cv2.putText(image, f'Smoothed bite probability: {round(smoothed_bite_probability * 100, 1)}%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if is_bite else (255, 0, 0), 2)
    cv2.imshow('Image', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
