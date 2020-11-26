import os
import random 
import numpy as np


import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras import layers



def make_model():
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(227,227,3)))
    
    model.add(layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu',input_shape=(227,227,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))

    model.add(layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))

    model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    
  
    model.add(layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation ='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation ='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation ='softmax'))
    
    return model


# Increase pooling layers 
def modification_1():
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(227,227,3)))
    
    model.add(layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu',input_shape=(227,227,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))

    model.add(layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    # Added
    model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))

    model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    # Added 
    model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))
  
    model.add(layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))
    
    model.add(layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation ='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation ='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation ='softmax'))

    return model 

# Increase network depth 
def modification_2():
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(227,227,3)))
    
    model.add(layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu',input_shape=(227,227,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))

    model.add(layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))

    model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())

    # added
    model.add(layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    
    # added
    model.add(layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))


    model.add(layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation ='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation ='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation ='softmax'))
    
    return model

