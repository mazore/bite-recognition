from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
import cv2

model = MobileNetV2(weights='imagenet')

data = np.empty((1, 224, 224, 3))
data[0] = cv2.imread('dog.jpg')
data = preprocess_input(data)
print(data)
predictions = model.predict(data)
for name, desc, score in decode_predictions(predictions)[0]:
    print('- {} ({:.2f}%%)'.format(desc, 100 * score))
