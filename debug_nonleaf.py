import cv2
import numpy as np
from PIL import Image

def is_probably_not_leaf(img):
    rgb = np.array(img.convert('RGB'), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    std = float(np.std(gray))
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))
    ptp = float(np.ptp(rgb, axis=(0, 1)).mean())
    print('std', std, 'edge_density', edge_density, 'ptp', ptp)
    if std < 12.0 and edge_density < 0.03:
        return True
    if std < 25.0 and edge_density < 0.08 and ptp > 180:
        return True
    return False

img = Image.open('dataset/Sirih_layu/20260419_131019.jpg')
print(is_probably_not_leaf(img))
