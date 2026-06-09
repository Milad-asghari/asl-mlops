import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
from zenml import step

@step
def tts(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    return X_train, X_test, y_train, y_test
