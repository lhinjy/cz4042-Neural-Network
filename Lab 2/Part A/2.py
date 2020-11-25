#
# Project 2, starter code Part a
#

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

# Create folder to store models and results
save_path = ('2')
if not os.path.isdir(save_path):
    os.makedirs(save_path)
if not os.path.exists('%s/models'%(save_path)):
    os.mkdir('%s/models'%(save_path))
if not os.path.exists('./%s/results'%(save_path)):
    os.mkdir('%s/results'%(save_path))

# This is required when using GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# Fixed, no need change
def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  # python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32) / 255
    labels = np.array(labels, dtype=np.int32)
    return data, labels


def make_model(num_ch_c1, num_ch_c2, use_dropout):

    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(3072, )))
    model.add(layers.Reshape(target_shape=(32, 32, 3), input_shape=(3072,)))
    
    model.add(layers.Conv2D(num_ch_c1, 9, activation='relu', input_shape=(None, None, 3)))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='VALID'))
    
    model.add(layers.Conv2D(num_ch_c2, 5, activation='relu', input_shape=(None, None, 3)))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='VALID'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(300))
    model.add(layers.Dense(10, use_bias=True, input_shape=(300,)))  # Here no softmax because we have combined it with the loss
    return model



def main():
    acc_=[]
    acc_index = 0
    val_acc = 0.0
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)

    num_ch_c1 = [10,30,50,70,90]  # Question 2
    num_ch_c2 = [20,40,60,80,100]  # Question 2

    epochs = 1000  # Fixed
    batch_size = 128  # Fixed
    learning_rate = 0.001
    optimizer_ = 'SGD'  # Question 3
    use_dropout = False  # Question 3(d) (see make_model)
    

    for i in (num_ch_c1):
        for d in (num_ch_c2):
            model = make_model(i, d, use_dropout)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            x_train, y_train = load_data('data_batch_1')
            x_test, y_test = load_data('test_batch_trim')


            model.compile(optimizer='sgd', loss=loss, metrics='accuracy')
            history = model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=0,
                validation_data=(x_test, y_test))

            # Accuracy of all epochs 
            train_acc = history.history['accuracy']
            test_acc = history.history['val_accuracy']

            # Accuracy of the last epoch
            acc_.append(history.history['val_accuracy'][epochs-1])

            print(history.history['val_accuracy'])

            print ("num_ch_c1: " +str(i) + " num_ch_c2: " + str(d) + " test_acc = " + str(acc_[acc_index]) )
            print("acc+" + str(acc_))
            acc_index = acc_index + 1

            plt.plot(range(1, len(train_acc) + 1), train_acc, label='Train')
            plt.plot(range(1, len(train_acc) + 1), test_acc, label='Test')
            plt.title('Model Accuracy: c1={i}, c2={d}')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(loc = 'best')
            plt.savefig(f'./{save_path}/results/num_ch_c1_{i}_num_ch_c2_{d}_{optimizer_}_no_dropout_accuracy.png')
            plt.show()
            plt.close()



if __name__ == '__main__':
    main()
