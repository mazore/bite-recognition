import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your model
model = load_model('bite_action_classifier_model.h5')

# Load the image with OpenCV
image_path = 'data/test/no-bite/563.jpg'
image = cv2.imread(image_path)

# Resize the image to the size the model expects (e.g., 224x224 for MobileNetV2)
input_size = (224, 224)
resized_image = cv2.resize(image, input_size)

# Normalize the image (scale pixel values to [0, 1])
normalized_image = resized_image / 255.0

# Add a batch dimension since the model expects batches of images
input_image = np.expand_dims(normalized_image, axis=0)

# Run the model on the input image
prediction = model.predict(input_image)

# Since this is a binary classifier, the output will be a single value
# Apply a threshold to get a binary result
bite_threshold = 0.5
is_bite = prediction[0][0] >= bite_threshold

# Print the result
print(f"Bite detected: {is_bite}")
