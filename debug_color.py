import os
import cv2
import numpy as np
from PIL import Image

for folder in ["dataset/Sirih_layu", "dataset/Sirih_sakit", "dataset/Sirih_sehat"]:
    files = [os.path.join(folder, f) for f in os.listdir(folder)[:20]]
    hues = []
    sats = []
    vals = []
    for path in files:
        img = np.array(Image.open(path).convert("RGB"))
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hues.append(hsv[:, :, 0].mean())
        sats.append(hsv[:, :, 1].mean())
        vals.append(hsv[:, :, 2].mean())
    print(folder)
    print(" mean_hue", round(float(np.mean(hues)), 2), "mean_sat", round(float(np.mean(sats)), 2), "mean_val", round(float(np.mean(vals)), 2))
    print(" std_hue", round(float(np.std(hues)), 2), "std_sat", round(float(np.std(sats)), 2), "std_val", round(float(np.std(vals)), 2))
