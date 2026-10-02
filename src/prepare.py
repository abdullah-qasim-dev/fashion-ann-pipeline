import os
import numpy as np
from tensorflow.keras.datasets import fashion_mnist


def main():
    os.makedirs("data/raw", exist_ok=True)

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    np.save("data/raw/x_train.npy", x_train)
    np.save("data/raw/y_train.npy", y_train)
    np.save("data/raw/x_test.npy", x_test)
    np.save("data/raw/y_test.npy", y_test)

    print("Fashion-MNIST raw data saved successfully.")
    print("Training images:", x_train.shape)
    print("Training labels:", y_train.shape)
    print("Test images:", x_test.shape)
    print("Test labels:", y_test.shape)


if __name__ == "__main__":
    main()