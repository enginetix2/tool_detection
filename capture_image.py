import cv2
import time
import os

# Initialize the webcam
cap = cv2.VideoCapture(1)  # Use 0 for the default camera

image_count = 5  # Number of images to capture
save_path = "shadowbox_images/"  # Folder to save images
os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists

try:
    for i in range(image_count):
        ret, frame = cap.read()
        if ret:
            filename = f"{save_path}shadowbox_{i}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            time.sleep(2)  # Wait 2 seconds before capturing the next image
finally:
    cap.release()
    cv2.destroyAllWindows()
cv2.destroyAllWindows()
