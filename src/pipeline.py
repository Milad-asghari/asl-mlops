from data_loader import load_data
from preprocessing import preprocess
from split import tts
from train import build_train
from zenml import pipeline


@pipeline
def tf_asl(data_dir: str, image_size: tuple):

    images, labels = load_data(data_dir=data_dir, image_size=image_size)
    X, y, le = preprocess(images = images, labels = labels)
    X_train, X_test, y_train, y_test = tts(X = X, y = y)
    build_train(X_train, X_test, y_train, y_test, le)