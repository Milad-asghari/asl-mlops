import os
import numpy as np
from PIL import Image
from typing import Tuple
from zenml import step

@step
def load_data(data_dir: str, image_size: tuple) -> Tuple[np.ndarray, np.ndarray]:

    images = []
    labels = []

    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for image_file in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_file)
                
                image = Image.open(image_path).convert("RGB")
                image = image.resize(image_size)
                images.append(np.array(image))
                labels.append(label)
                
    images = np.array(images)
    labels = np.array(labels)    
    return images, labels