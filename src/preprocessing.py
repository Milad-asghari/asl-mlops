import numpy as np
from sklearn.preprocessing import LabelEncoder

from typing import Tuple
from zenml import step


@step
def preprocess(images: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    X = images / 255.0
    le = LabelEncoder()
    y = le.fit_transform(labels)

    return X, y, le