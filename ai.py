import os
os.environ["KERAS_BACKEND"] = "jax"

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.preprocessing.image import load_img, img_to_array


IMAGE_SIZE = 64
STEPS_PER_SGRAM = 20


def load_single_sgram(sgram_image_path):
    """
    Load the data from a single spectrogram image in a format useful for a neural network.
    """
    # this reads the file, resizes it, makes it grayscale, and scales the inputs to values
    # between 0 and 1
    return img_to_array(
        load_img(
            sgram_image_path,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            color_mode="grayscale",
        )
    ) / 255


def load_sgrams_and_steps(sgrams_images_path, steps_csv_path):
    """
    Load the input spectrogram images and output steps from the files.
    Returns two numpy arrays: sgrams and steps.
    Sgrams (spectrograms) are the inputs for the neural network, while steps are the outputs.
    """
    sorted_images = list(sorted(Path(sgrams_images_path).glob("*.jpg")))

    print("Reading spectrogram images...")
    sgrams = np.array([
        load_single_sgram(image_path)
        for image_path in sorted_images
    ])

    # TODO load outputs from a csv, this is just an example using fake data
    steps = np.array([
        ([1] * STEPS_PER_SGRAM) if "pared" in image_path.name else ([0] * STEPS_PER_SGRAM)
        for image_path in sorted_images
    ])

    return sgrams, steps


def show_sgrams(samples):
    """
    Display some sgram images, useful to use inside jupyter notebooks.
    """
    for sample in samples:
        plt.axis('off')
        plt.imshow(sample)
        plt.show()


def build_and_train_neural_network(sgrams, steps, test_split=0.2, train_epochs=5):
    """
    Build a convolutional neural network that is able to predict steps from spectrograms.
    """
    train_sgrams, test_sgrams, train_steps, test_steps = train_test_split(
        sgrams, steps, test_size=test_split,
    )

    neural_network = Sequential([
        Input((IMAGE_SIZE, IMAGE_SIZE, 1)),

        Convolution2D(filters=10, kernel_size=(4, 4), strides=1, activation="relu"),
        Dropout(0.25),

        Convolution2D(filters=10, kernel_size=(4, 4), strides=1, activation="relu"),
        Dropout(0.25),

        MaxPooling2D(pool_size=(4, 4)),

        Flatten(),

        Dense(100, activation="tanh"),
        Dropout(0.25),

        Dense(STEPS_PER_SGRAM, activation="sigmoid"),
    ])

    neural_network.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy",],
    )

    neural_network.fit(
        train_sgrams,
        train_steps,
        epochs=train_epochs,
        batch_size=128,
        validation_data=(test_sgrams, test_steps),
    )

    return neural_network


def predict(neural_network, sgram_path):
    """
    Load a spectrogram and predict its steps. Returns both the final and the raw predictions.
    Raw predictions are only useful for debugging, while the final predictions are the steps
    we should use for the game.
    """
    inputs = np.array([load_single_sgram(sgram_path)])  # armamos un "dataset" con solo esa imagen
    predictions = neural_network.predict(inputs)
    return predictions.round(), predictions


def show_and_predict(neural_network, sgram_path):
    """
    Utility to test predictions in a jupyter notebook, showing the spectrogram and the predicted
    steps.
    """
    inputs = np.array([load_single_sgram(sgram_path)])
    show_sgrams(inputs)

    predictions, raw_predictions = predict(neural_network, sgram_path)
    print("Prediction:")
    print(predictions)
    print("Raw prediction:")
    print(raw_predictions)
