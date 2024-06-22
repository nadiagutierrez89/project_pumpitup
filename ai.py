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


class PumpIA:
    def __init__(self):
        self.reduce_images_to = 64, 64

        self.sgrams_root_path = None
        self.dataset_json_path = None
        self.sgrams = None
        self.steps = None
        self.output_steps_per_sgram = None

        self.neural_network = None

    def load_single_sgram(self, sgram_image_path):
        """
        Load the data from a single spectrogram image in a format useful for a neural network.
        """
        # this reads the file, resizes it, makes it grayscale, and scales the inputs to values
        # between 0 and 1
        return img_to_array(
            load_img(
                sgram_image_path,
                target_size=self.reduce_images_to,
                color_mode="grayscale",
            )
        ) / 255

    def load_dataset(self, dataset_json_path, train_sgrams_root_path):
        """
        Load the input spectrogram images and output steps from the files.
        Sgrams (spectrograms) are the inputs for the neural network, while steps are the outputs.
        """
        self.dataset_json_path = Path(dataset_json_path)
        self.train_sgrams_root_path = Path(train_sgrams_root_path)

        with open(dataset_json_path, "r") as dataset_file:
            dataset_config = json.load(dataset_file)

        # sort them to then generate inputs and outputs in synced order
        sorted_sgrams = list(sorted(
            sgram_name
            for sgram_name in dataset_config["sgrams_with_steps"]
        ))

        print("Reading spectrogram images...")
        self.sgrams = np.array([
            self.load_single_sgram(self.train_sgrams_root_path / sgram_name)
            for sgram_name in sorted_sgrams
        ])

        self.steps = np.array([
            [int(digit)
             for digit in dataset_config["sgrams_with_steps"][sgram_name]]
            for sgram_name in sorted_sgrams
        ])
        print("Loaded", len(self.sgrams), "spectrograms")

        self.output_steps_per_sgram = dataset_config["output_steps_per_sgram"]

    def show_sgrams(self, sgrams=None, sample=None):
        """
        Display some sgram images, useful to use inside jupyter notebooks.
        """
        if sgrams is None:
            sgrams = self.sgrams

        if sample is not None:
            sgrams = sgrams[np.random.randint(sgrams.shape[0], size=sample), :]

        for sgram in sgrams:
            plt.axis('off')
            plt.imshow(sgram)
            plt.show()

    def build_and_train_neural_network(self, test_split=0.2, train_epochs=5):
        """
        Build a convolutional neural network that is able to predict steps from spectrograms.
        """
        train_sgrams, test_sgrams, train_steps, test_steps = train_test_split(
            self.sgrams, self.steps, test_size=test_split,
        )

        self.neural_network = Sequential([
            Input((*self.reduce_images_to, 1)),

            Convolution2D(filters=10, kernel_size=(4, 4), strides=1, activation="relu"),
            Dropout(0.25),

            Convolution2D(filters=10, kernel_size=(4, 4), strides=1, activation="relu"),
            Dropout(0.25),

            MaxPooling2D(pool_size=(4, 4)),

            Flatten(),

            Dense(100, activation="tanh"),
            Dropout(0.25),

            Dense(self.output_steps_per_sgram, activation="sigmoid"),
        ])

        self.neural_network.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["binary_accuracy",],
        )

        self.neural_network.fit(
            train_sgrams,
            train_steps,
            epochs=train_epochs,
            batch_size=128,
            validation_data=(test_sgrams, test_steps),
        )

    def predict(self, sgram_path, raw=False):
        """
        Load a spectrogram and predict its steps. If raw=True, the raw not-rounded predictions
        are returned.
        Raw predictions are only useful for debugging, while the final predictions are the steps
        we should use for the game.
        """
        inputs = np.array([self.load_single_sgram(sgram_path)])  # un "dataset" con solo esa imagen
        predictions = self.neural_network.predict(inputs)[0]
        if not raw:
            predictions = "".join(str(digit) for digit in predictions.round().astype(int))

        return predictions


    def show_and_predict(self, sgram_path):
        """
        Utility to test predictions in a jupyter notebook, showing the spectrogram and the predicted
        steps.
        """
        inputs = np.array([self.load_single_sgram(sgram_path)])
        self.show_sgrams(inputs)

        predictions = self.predict(sgram_path)
        raw_predictions = self.predict(sgram_path, raw=True)
        print("Prediction:")
        print(predictions)
        print("Raw prediction:")
        print(raw_predictions)
