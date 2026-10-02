import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf


def main():
    model = tf.keras.models.load_model("models/model.h5")

    x_test = np.load("data/processed/x_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    display.plot()
    plt.title("Fashion-MNIST Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy)
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)
    print("Metrics saved to metrics.json")
    print("Confusion matrix saved to confusion_matrix.png")


if __name__ == "__main__":
    main()