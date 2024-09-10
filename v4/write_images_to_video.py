import cv2
import numpy as np

# Example: Assuming images is a numpy array of shape (num_frames, height, width, channels)
# Where num_frames is the number of frames, height is the height of each frame, width is the width of each frame,
# and channels is 3 for RGB images.


def save_images(images, filename):
    # Define the codec and create VideoWriter object
    # 'XVID' is a commonly used codec. You can replace it with 'MJPG', 'X264', etc.
    # Use 'mp4v' codec for MP4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_file = f'{filename}.mp4'  # Output video file name
    fps = 20.0  # Frames per second
    height, width, channels = images[0].shape

    # Initialize the video writer
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Loop through each image and write it to the video
    for image in images:
        video_writer.write(image)

    # Release the video writer object
    video_writer.release()


images = np.random.randint(0, 255, (100, 480, 640, 3), dtype=np.uint8)  # Replace with your actual images array
save_images(images)
print(f"Video saved as {output_file}")
