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
save_path = ('3d')
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
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, use_bias=True, input_shape=(300,)))  # Here no softmax because we have combined it with the loss
    return model



def main():
    val_acc = 0.0
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)

    num_ch_c1 = 70  # Question 2
    num_ch_c2 = 80  # Question 2

    epochs = 1000  # Fixed
    batch_size = 128  # Fixed
    learning_rate = 0.001
    optimizer_ = 'SGD'  # Question 3
    use_dropout = True  # Question 3(d) (see make_model)

    model = make_model(num_ch_c1, num_ch_c2, use_dropout)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if optimizer_ == 'SGD':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_ == 'SGD-momentum':  # Question 3(a)
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    elif optimizer_ == 'RMSProp':  # Question 3(b)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_ == 'Adam':  # Question 3(c)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise NotImplementedError(f'You do not need to handle [{optimizer_}] in this project.')

    # Training and test
    x_train, y_train = load_data('data_batch_1')
    x_test, y_test = load_data('test_batch_trim')

    # Training
    model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test))

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    test_acc = history.history['val_accuracy']
    print(history.history['val_accuracy'])

    # Save model
    model.save(f'./models/{num_ch_c1}_{num_ch_c2}_{optimizer_}_dropout')



    # Save the plot for losses
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Test')
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.savefig(f'./{save_path}/results/{num_ch_c1}_{num_ch_c2}_{optimizer_}_dropout_loss.pdf')
    plt.close()
    plt.show()

    # Save the plot for accuracies
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='Train')
    plt.plot(range(1, len(test_acc) + 1), test_acc, label='Test')
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc = 'best')
    plt.savefig(f'./{save_path}/results/{num_ch_c1}_{num_ch_c2}_{optimizer_}_dropout_accuracy.pdf')
    plt.close()
    plt.show()


if __name__ == '__main__':
    main()
