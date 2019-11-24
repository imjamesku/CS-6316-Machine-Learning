# -*- coding: utf-8 -*-
"""fashion_template.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/gist/jamesku1996/05755377a54d1b095b216f051131e19b/fashion_template.ipynb

Upload the fashion_train.csv file to your google drive and specify the correct path in the main method. When prompted, provide the authorization key.
"""

# Machine Learning Homework 4 - Image Classification

__author__ = '**'

# General imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import os
import sys
import pandas as pd

# Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import BatchNormalization

# Google Colab stuff
# from google.colab import drive
# drive.mount('/content/drive')

"""The below methods have been provided for you."""

# Already implemented


def get_data(datafile):
    dataframe = pd.read_csv(datafile)
    dataframe = shuffle(dataframe)
    data = list(dataframe.values)
    labels, images = [], []
    for line in data:
        labels.append(line[0])
        images.append(line[1:])
    labels = np.array(labels)
    images = np.array(images).astype('float32')
    images /= 255
    return images, labels


# Already implemented
def visualize_weights(trained_model, num_to_display=20, save=True, hot=True):
    layer1 = trained_model.layers[0]
    weights = layer1.get_weights()[0]

    # Feel free to change the color scheme
    colors = 'hot' if hot else 'binary'

    for i in range(num_to_display):
        wi = weights[:, i].reshape(28, 28)
        plt.imshow(wi, cmap=colors, interpolation='nearest')
        plt.show()


# Already implemented
def output_predictions(predictions):
    with open('predictions.txt', 'w+') as f:
        for pred in predictions:
            f.write(str(pred) + '\n')


"""Implement the following method to generate plots of the train and validation accuracy and loss vs epochs. 
You should call this method for your best-performing MLP model and best-performing CNN model 
(4 plots total--2 accuracy plots, 2 loss plots).
"""


def plot_history(history):
    train_loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']

    train_acc_history = history.history['acc']
    val_acc_history = history.history['val_acc']

    # plot
    print(train_loss_history)
    plt.title('training loss & validation loss')
    plt.plot(list(range(len(train_loss_history))),
             train_loss_history, label='train loss history')
    plt.legend()
    plt.plot(list(range(len(val_loss_history))),
             val_loss_history, label='val_loss history')
    plt.legend()
    plt.show()

    plt.title('training accuracy & validation accuracy')
    plt.plot(list(range(len(train_acc_history))),
             train_acc_history, label='train acc histroy')
    plt.legend()
    plt.plot(list(range(len(val_acc_history))),
             val_acc_history, label='val acc history')
    plt.legend()

    plt.show()


"""Code for defining and training your MLP models"""


def create_mlp(args=None):
    # You can use args to pass parameter values to this method

    # Define model architecture
    model = Sequential()
    model.add(Dense(units=512, activation='relu', input_dim=28*28))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    # add more layers...
    model.summary()

    # Define Optimizer
    # optimizer = keras.optimizers.SGD(...)
    optimizer = keras.optimizers.RMSprop()

    # Compile
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

    return model


def train_mlp(x_train, y_train, x_vali=None, y_vali=None, args=None):
    # You can use args to pass parameter values to this method
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    model = create_mlp(args)
    history = model.fit(x_train, y_train,
                        batch_size=args['batch_size'],
                        epochs=args['epochs'],
                        verbose=1,
                        validation_split=args['validation_split'])
    return model, history


"""Code for defining and training your CNN models"""


def create_cnn(args=None):
    # You can use args to pass parameter values to this method

    # 28x28 images with 1 color channel
    input_shape = (28, 28, 1)
    num_classes = 10
    # Define model architecture
    model = Sequential()
    model.add(Conv2D(filters=32, activation='relu',
                     kernel_size=(3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, activation='relu', kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, activation='relu',
                     kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    # model.add(BatchNormalization())
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), activation='relu', strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    # can add more layers here...
    model.add(Flatten())
    # can add more layers here...
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    # Optimizer
    optimizer = optimizer = keras.optimizers.Adadelta()

    # Compile
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer, metrics=['accuracy'])

    return model


def train_cnn(x_train, y_train, x_vali=None, y_vali=None, args=None):
    # You can use args to pass parameter values to this method
    x_train = x_train.reshape(-1, 28, 28, 1)
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    model = create_cnn(args)
    history = model.fit(x_train, y_train,
                        batch_size=args['batch_size'],
                        epochs=args['epochs'],
                        verbose=1,
                        validation_split=args['validation_split'])
    return model, history


"""An optional method you can use to repeatedly call create_mlp, train_mlp, create_cnn, or train_cnn. 
You can use it for performing cross validation or parameter searching.
"""


def train_and_select_model(train_csv):
    """Optional method. You can write code here to perform a 
    parameter search, cross-validation, etc. """

    x_train, y_train = get_data(train_csv)

    args = {
        'learning_rate': 0.01,
        'batch_size': 128,
        'epochs': 35,
        'validation_split': 0.1
    }
    # model, history = train_mlp(x_train, y_train, x_vali=None, y_vali=None, args=args)
    model, history = train_cnn(
        x_train, y_train, x_vali=None, y_vali=None, args=args)
    validation_accuracy = history.history['val_acc']

    return model, history


"""Main method. Make sure the file paths here point to the correct place in your google drive."""

if __name__ == '__main__':
    ### Switch to "development_mode = False" before you submit ###
    grading_mode = True
    if grading_mode:
        # When we grade, we'll provide the file names as command-line arguments
        if (len(sys.argv) != 3):
            print("Usage:\n\tpython3 fashion.py train_file test_file")
            exit()
        train_file, test_file = sys.argv[1], sys.argv[2]
        x_train, y_train = get_data(train_file)
        x_test, y_test = get_data(test_file)
        x_test = x_test.reshape(-1, 28, 28, 1)

        # train your best model
        args = {
            'batch_size': 128,
            'epochs': 35,
            'validation_split': 0.0
        }
        best_model, history = train_cnn(
            x_train, y_train, x_vali=None, y_vali=None, args=args)

        # use your best model to generate predictions for the test_file
        predictions = best_model.predict(x_test)
        predictions = [np.argmax(prediction) for prediction in predictions]
        print(type(predictions))
        output_predictions(predictions)

    # Include all of the required figures in your report. Don't generate them here.
    else:
        # Edit the following two lines if your paths are different
        train_file = '/content/drive/My Drive/fashion_data/fashion_train.csv'
        test_file = '/content/drive/My Drive/fashion_data/fashion_test_labeled.csv'
        x_train, y_train = get_data(train_file)
        mlp_model, mlp_history = train_and_select_model(train_file)
        plot_history(mlp_history)
        visualize_weights(mlp_model)
