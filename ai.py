import os
os.environ["KERAS_BACKEND"] = "jax"

import json
from collections import namedtuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.preprocessing.image import load_img, img_to_array


Dataset = namedtuple("Dataset", "sgrams steps output_steps_per_sgram")


SGRAMS_ROOT = Path("./sgrams")
REDUCE_IMAGES_TO = 64, 64


def load_single_sgram(sgram_image_path):
    """
    Load the data from a single spectrogram image in a format useful for a neural network.
    """
    # this reads the file, resizes it, makes it grayscale, and scales the inputs to values
    # between 0 and 1
    return img_to_array(
        load_img(
            sgram_image_path,
            target_size=(REDUCE_IMAGES_TO[0], REDUCE_IMAGES_TO[1]),
            color_mode="grayscale",
        )
    ) / 255


def load_dataset(dataset_path):
    """
    Load the input spectrogram images and output steps from the files.
    Returns two numpy arrays: sgrams and steps.
    Sgrams (spectrograms) are the inputs for the neural network, while steps are the outputs.
    """
    with open(dataset_path, "r") as dataset_file:
        dataset_config = json.load(dataset_file)

    # sort them to then generate inputs and outputs in synced order
    sorted_sgrams = list(sorted(
        sgram_name
        for sgram_name in dataset_config["sgrams_with_steps"]
    ))

    print("Reading spectrogram images...")
    sgrams = np.array([
        load_single_sgram(SGRAMS_ROOT / sgram_name)
        for sgram_name in sorted_sgrams
    ])

    steps = np.array([
        [int(digit)
         for digit in dataset_config["sgrams_with_steps"][sgram_name]]
        for sgram_name in sorted_sgrams
    ])

    return Dataset(sgrams, steps, dataset_config["output_steps_per_sgram"])


def show_sgrams(sgrams, sample=None):
    """
    Display some sgram images, useful to use inside jupyter notebooks.
    """
    if sample is not None:
        sgrams = sgrams[np.random.randint(sgrams.shape[0], size=sample), :]

    for sgram in sgrams:
        plt.axis('off')
        plt.imshow(sgram)
        plt.show()


def build_and_train_neural_network(dataset, test_split=0.2, train_epochs=5):
    """
    Build a convolutional neural network that is able to predict steps from spectrograms.
    """
    train_sgrams, test_sgrams, train_steps, test_steps = train_test_split(
        dataset.sgrams, dataset.steps, test_size=test_split,
    )

    neural_network = Sequential([
        Input((REDUCE_IMAGES_TO[0], REDUCE_IMAGES_TO[1], 1)),

        Convolution2D(filters=10, kernel_size=(4, 4), strides=1, activation="relu"),
        Dropout(0.25),

        Convolution2D(filters=10, kernel_size=(4, 4), strides=1, activation="relu"),
        Dropout(0.25),

        MaxPooling2D(pool_size=(4, 4)),

        Flatten(),

        Dense(100, activation="tanh"),
        Dropout(0.25),

        Dense(dataset.output_steps_per_sgram, activation="sigmoid"),
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
