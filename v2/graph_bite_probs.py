import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
from extract_face import extract_face
from tensorflow.keras.models import load_model

model = load_model('training3.h5')
input_size = (224, 224)
bite_probability_deque = deque(maxlen=3)

def get_bite_probabilities(image):
    image = cv2.flip(image, 1)

    extracted_face = extract_face(image)
    if extracted_face is None or extracted_face.shape[0] == 0 or extracted_face.shape[1] == 0:
        return 0, 0, image

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

    return bite_probability, smoothed_bite_probability, image



# Initialize video capture (0 for the first webcam, or provide a video file path)
cap = cv2.VideoCapture(0)

# Initialize the figure and axis for plotting
fig, ax = plt.subplots()
x_data, y1_data, y2_data = [], [], []
ln1, = ax.plot([], [], 'r-', label='y1')
ln2, = ax.plot([], [], 'b-', label='y2')
ax.set_xlim(0, 100)
ax.set_ylim(-10, 10)
ax.legend()

# Function to initialize the plot
def init():
    ln1.set_data([], [])
    ln2.set_data([], [])
    return ln1, ln2

# Function to update the plot for each frame
def update(frame_count):
    ret, frame = cap.read()
    if not ret:
        return ln1, ln2

    # Your logic to calculate y1 and y2 based on the frame
    y1, y2, frame = get_bite_probabilities(frame)

    x_data.append(frame_count)
    y1_data.append(y1 * 100)
    y2_data.append(y2 * 100)

    ln1.set_data(x_data, y1_data)
    ln2.set_data(x_data, y2_data)

    # Limit x and y data to the last 100 points for performance
    if len(x_data) > 20:
        x_data.pop(0)
        y1_data.pop(0)
        y2_data.pop(0)

    ax.set_xlim(max(0, frame_count - 20), frame_count)
    ax.set_ylim(min(min(y1_data), min(y2_data)) - 1, max(max(y1_data), max(y2_data)) + 1)

    # Display the video frame in a separate window
    cv2.imshow('Video Feed', frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        plt.close()

    return ln1, ln2

frame_count = 0

# Create the animation object
ani = FuncAnimation(fig, update, frames=np.arange(0, 200), init_func=init, blit=True)

plt.show()
