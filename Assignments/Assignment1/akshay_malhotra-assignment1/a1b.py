# Extension used with respect to tensorboard.
# %load_ext tensorboard

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import datetime


# Logs and callbacks linked to generate tensorboard graph
# logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

# Function responsible for importing MNIST Dataset
def import_data():
    return tf.keras.datasets.mnist.load_data()


# Function responsible for cleaning data (pre-processing)
def clean_data(x_train, y_train, x_test, y_test):

    # Normalizing the data
    x_train=  x_train / 255.0
    x_test = x_test / 255.0

    # Reshaping the training and testing data
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    # One hot encoding for output labels
    y_train = tf.one_hot(y_train, depth=10)
    y_test = tf.one_hot(y_test, depth=10)

    return x_train, y_train, x_test, y_test


# Function responsible for generating CNN model
def generate_cnn_model():
    filter_dims = (3,3)

    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, filter_dims, strides=1, padding='same', activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),

            tf.keras.layers.Conv2D(64, filter_dims, strides=1, padding='same', activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax")
        ]
    )


(X_train, Y_train), (X_test, Y_test) = import_data()
X_train, Y_train, X_test, Y_test = clean_data(X_train, Y_train, X_test, Y_test)
model = generate_cnn_model()

# Prints the model summary
print ("Model Summary")
print (model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=15)
# model.fit(X_train, Y_train, epochs=15, callbacks=[tensorboard_callback])
test_loss, test_acc = model.evaluate(X_test, Y_test)

# %tensorboard --logdir logs