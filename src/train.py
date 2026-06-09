import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from tensorflow.keras import losses, optimizers, callbacks
from typing import Tuple
from zenml import step
import mlflow
from model import build_cnn

@step
def build_train(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, le: LabelEncoder):
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("cnn_model_experiment")

    model = build_cnn(
    input_shape=X_train.shape[1:],
    num_classes=len(le.classes_)
    
    )
    adam = optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=adam,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    
    es = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    with mlflow.start_run(run_name="cnn_model_run"):
        history = model.fit(
            X_train, y_train,
            validation_split = 0.2,
            epochs=200,
            batch_size=32,
            callbacks=[es, rlr],
            verbose=0
        )

        training_loss = model.evaluate(X_train, y_train, verbose=0)[0]
        training_accuracy = model.evaluate(X_train, y_train, verbose=0)[1]

        test_loss = model.evaluate(X_test, y_test, verbose=0)[0]
        test_accuracy = model.evaluate(X_test, y_test, verbose=0)[1]

        mlflow.log_metric("training_loss", training_loss)
        mlflow.log_metric("training_accuracy", training_accuracy)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)

        mlflow.tensorflow.log_model(model, 'tf_asl_model')

        plt.figure(figsize=(10,10))

        plt.plot(history.history['loss'], label = 'loss')
        plt.plot(history.history['val_loss'], label = 'val_loss')

        plt.xlabel('epochs')
        plt.ylabel('sparse_categorical_crossentropy')
        plt.legend()
        plt.yscale('log')
        
        plt.savefig('loss_plot.png')
        plt.close()

        mlflow.log_artifact('loss_plot.png')

        y_pred = model.predict(X_test, verbose = 0)
        y_pred_argmax = np.argmax(y_pred, axis = 1)
        cm = confusion_matrix(y_test, y_pred_argmax)
        sns.heatmap(cm, annot=True, xticklabels=le.classes_, yticklabels=le.classes_)
        plt.savefig('confusion_matrix.png')
        plt.close()

        mlflow.log_artifact('confusion_matrix.png')
