import os
import random 
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def make_levi():
  model = tf.keras.Sequential()
  model.add(layers.Input(shape=(227,227,3)))

  model.add(layers.Conv2D(filters=96, kernel_size=(7,7), activation='relu',input_shape=(227,227,3)))
  model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))
  model.add(layers.BatchNormalization())

  model.add(layers.Conv2D(filters=256, kernel_size=(5,5), activation='relu', padding="same"))
  model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))
  model.add(layers.BatchNormalization())

  model.add(layers.Conv2D(filters=384, kernel_size=(3,3), activation='relu', padding="same"))
  model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))

  model.add(layers.Flatten())
  model.add(layers.Dense(512, activation ='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(512, activation ='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(10, activation ='softmax'))

  return model


# increase drop out value and drop out layer by 0.1/1
def modification_1():
  model = tf.keras.Sequential()
  model.add(layers.Input(shape=(227,227,3)))
  model.add(layers.Conv2D(filters=96, kernel_size=(7,7), activation='relu',input_shape=(227,227,3)))
  model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))
  model.add(layers.BatchNormalization())

  model.add(layers.Conv2D(filters=256, kernel_size=(5,5), activation='relu', padding="same"))
  model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))
  model.add(layers.BatchNormalization())

  model.add(layers.Conv2D(filters=384, kernel_size=(3,3), activation='relu', padding="same"))
  model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))

  model.add(layers.Flatten())
  # added 
  model.add(layers.Dense(512, activation ='relu'))
  model.add(layers.Dropout(0.6))

  model.add(layers.Dense(512, activation ='relu'))
  # added
  model.add(layers.Dropout(0.6))
  model.add(layers.Dense(512, activation ='relu'))
  # added
  model.add(layers.Dropout(0.6))
  model.add(layers.Dense(10, activation ='softmax'))

  return model


# Increase convolution layer
def modification_2():
  model = tf.keras.Sequential()
  model.add(layers.Input(shape=(227,227,3)))
  model.add(layers.Conv2D(filters=96, kernel_size=(7,7), activation='relu',input_shape=(227,227,3)))
  model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))
  model.add(layers.BatchNormalization())
  
  model.add(layers.Conv2D(filters=256, kernel_size=(5,5), activation='relu', padding="same"))
  model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))
  model.add(layers.BatchNormalization())

  model.add(layers.Conv2D(filters=384, kernel_size=(3,3), activation='relu', padding="same"))
  model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))

  # added 
  model.add(layers.Conv2D(filters=256, kernel_size=(5,5), activation='relu', padding="same"))
  model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))
  model.add(layers.BatchNormalization())

  # added
  model.add(layers.Conv2D(filters=256, kernel_size=(5,5), activation='relu', padding="same"))
  model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))
  model.add(layers.BatchNormalization())

  model.add(layers.Flatten())
  model.add(layers.Dense(512, activation ='relu'))
  model.add(layers.Dropout(0.5))

  model.add(layers.Dense(512, activation ='relu'))
  model.add(layers.Dropout(0.5))

  model.add(layers.Dense(10, activation ='softmax'))
  return model


  





