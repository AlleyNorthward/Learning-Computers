import cv2
from PIL import Image
from pathlib import Path
import numpy as np
import os

IMAGE_DIR = Path(__file__).resolve().parent / "dataset" / "bmp" / "s1"

IMAGE_PATH = os.path.join(str(IMAGE_DIR), "1.BMP")
print(IMAGE_PATH)
image = Image.open(IMAGE_PATH).convert('L')
image_np = np.array(image)
print(image_np)
image_cv2 = cv2.imread(IMAGE_PATH, 0)
print(image_cv2)


