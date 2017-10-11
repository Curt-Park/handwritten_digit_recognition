import numpy as np

def normalize_images(images):
    numerator = images - np.expand_dims(np.mean(images, 1), 1)
    denominator = np.expand_dims(np.std(images, 1), 1)
    return numerator / (denominator + 1e-7)
