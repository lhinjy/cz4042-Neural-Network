import pandas as pd
import numpy as np
import tensorflow.keras as keras
import os
from pathlib import Path
from PIL import Image
import csv
import sys
import h5py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 227, 227
batch_size = 32
base_path = "C:/Users/User/jupyter/project/face_age_gender/resources/"
#cwd = os.getcwd()+"/resources/"
#base_path = cwd

datagen = ImageDataGenerator(
    rotation_range=6,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1)

# create path to get to data
def create_path(df, base_path):

    df['path'] = df.apply(lambda x: base_path+"aligned/"+x['user_id']+"/landmark_aligned_face.%s.%s"
                                                                      %(x['face_id'], x['original_image']), axis=1)

    return df

def read_and_resize(filepath, input_shape=(img_width, img_height)):
    im = Image.open((filepath)).convert('RGB')
    im = im.resize(input_shape)
    im_array = np.array(im, dtype="uint8")#[..., ::-1]
    return np.array(im_array / (np.max(im_array)+ 0.001), dtype="float32")

def augment(im_array):
    im_array = datagen.random_transform(im_array)
    return im_array

# load the image and labels from the path name return array of image
def gen(df, batch_size=32, aug=False):
    df = df.sample(frac=1)
    while True:
        for i, batch in enumerate([df[i:i+batch_size] for i in range(0,df.shape[0],batch_size)]):
            #print([df[i:i+batch_size] for i in range(0,df.shape[0],batch_size)])
            if aug:
                images = np.array([augment(read_and_resize(file_path)) for file_path in batch.path.values])
            else:
                images = np.array([read_and_resize(file_path) for file_path in batch.path.values])

            labels = np.array([int(g=="m") for g in batch.gender.values])

            yield images, labels

def filter_df(df):

    df['f'] = df.gender.apply(lambda x: int(x in ['f', 'm']))
    df = df[df.f == 1]
    return df


def data():
    test_id = 4
    train_id = [0,1,2,3]
    train_df = pd.concat([pd.read_csv(base_path+"fold_%s_data.txt"%i, sep="\t") for i in train_id])
    test_df = pd.read_csv(base_path+"fold_%s_data.txt"%test_id, sep="\t")

    train_df = filter_df(train_df)
    test_df = filter_df(test_df)

    train_df = create_path(train_df, base_path =base_path)
    test_df = create_path(test_df, base_path =base_path)

    train_d = gen(train_df,aug = True)
    test_d = gen(test_df, aug = True)
 
    return train_d, test_d


if __name__ == "__main__":
    data()
