import cv2
import os

input_dir = 'extracted/no-bite'
output_dir = 'resized/no-bite'
target_size = (224, 224)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    img = cv2.imread(os.path.join(input_dir, filename))
    if img is not None:
        resized_img = cv2.resize(img, target_size)
        cv2.imwrite(os.path.join(output_dir, filename), resized_img)
        print(filename)
