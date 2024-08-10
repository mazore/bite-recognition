import cv2
import os
import random

input_dir = 'data/no-bite'
train_dir = 'data/train/no-bite'
test_dir = 'data/test/no-bite'
valid_dir = 'data/validate/no-bite'
target_size = (224, 224)

filesnames = os.listdir(input_dir)
random.shuffle(filesnames)
for i, filename in enumerate(filesnames):
    if i < 0.7 * len(filesnames):
        os.rename(os.path.join(input_dir, filename), os.path.join(train_dir, filename))
    elif i < 0.85 * len(filesnames):
        os.rename(os.path.join(input_dir, filename), os.path.join(test_dir, filename))
    else:
        os.rename(os.path.join(input_dir, filename), os.path.join(valid_dir, filename))
