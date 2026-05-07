import cv2
import numpy as np

IMG_SIZE = 224  # from your notebook

def preprocess_image(img):
    # Convert RGB → BGR (because OpenCV)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # IMPORTANT: no normalization (EfficientNet handles it)
    img = img.astype(np.float32)

    return img