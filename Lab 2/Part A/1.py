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
    val_acc = 0.0
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)

    num_ch_c1 = 50  # Question 2
    num_ch_c2 = 60  # Question 2

    epochs = 1000  # Fixed
    batch_size = 128  # Fixed
    learning_rate = 0.001
    optimizer_ = 'SGD'  # Question 3
    use_dropout = False  # Question 3(d) (see make_model)

    model = make_model(num_ch_c1, num_ch_c2, use_dropout)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if optimizer_ == 'SGD':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_ == 'SGD-momentum':  # Question 3(a)
        raise NotImplementedError('Complete it by yourself')
    elif optimizer_ == 'RMSProp':  # Question 3(b)
        raise NotImplementedError('Complete it by yourself')
    elif optimizer_ == 'Adam':  # Question 3(c)
        raise NotImplementedError('Complete it by yourself')
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


    if (val_acc<test_acc[-1]):
        model.save_weights('best_weights.hdf5')
        print('Saved model')
    
    # Create folder to store models and results
    save_path = ('1b')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.exists('%s/models'%(save_path)):
        os.mkdir('%s/models'%(save_path))
    if not os.path.exists('./%s/results'%(save_path)):
        os.mkdir('%s/results'%(save_path))

    # Save model
    if use_dropout:
        model.save(f'./models/{num_ch_c1}_{num_ch_c2}_{optimizer_}_dropout')
    else:
        model.save(f'./models/{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout')


    # Save the plot for losses
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Test')
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.savefig(f'./{save_path}/results/{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout_loss.pdf')
    plt.close()
    plt.show()

    # Save the plot for accuracies
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='Train')
    plt.plot(range(1, len(test_acc) + 1), test_acc, label='Test')
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc = 'best')
    plt.savefig(f'./{save_path}/results/{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout_accuracy.pdf')
    plt.close()
    plt.show()

    model.load_weights('best_weights.hdf5')
    model.save('shapes_cnn.h5')

    # Image 1
    image1 = x_test[0,:]
    image1_show = image1.reshape(3, 32, 32).transpose(1, 2, 0)
    plt.gray()
    plt.imshow(image1_show)
    plt.savefig('./%s/results/image1_original.png'%(save_path))

    # Image 2
    image2 = x_test[1,:]
    image2_show = image2.reshape(3, 32, 32).transpose(1, 2, 0)
    plt.imshow(image2_show)
    plt.savefig('./%s/results/image2_original.png'%(save_path))

    layer_outputs = [layer.output for layer in model.layers[:5]] 
    # Creates a model that will return these outputs, given the model input
    activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)

    # Format to fit 
    imag1 = np.expand_dims(image1, axis =0)
    activations1 = activation_model.predict(imag1) 

    imag2 = np.expand_dims(image2, axis =0)
    activations2 = activation_model.predict(imag2) 

    # image 1, c1
    img1_c1 = activations1[1]

    print(img1_c1.shape)
    for i in range(50):
        plt.subplot(5,10,i+1)
        plt.axis('off')
        plt.imshow(img1_c1[0, :, :, i])
    plt.suptitle('Image 1: Feature maps at conv1')
    plt.savefig('./%s/results/image1_c1.png'%(save_path))

    # image 1, s1
    img1_s1 = activations1[2]

    for i in range(50):
        plt.subplot(5,10,i+1)
        plt.axis('off')
        plt.imshow(img1_s1[0, :, :, i])
    plt.suptitle('Image 1: Feature map at pool1')
    plt.savefig('./%s/results/image1_s1.png'%(save_path))

    # image 1, c2
    imag1_c2 = activations1[3]

    for i in range(60):
        plt.subplot(6,10,i+1)
        plt.axis('off')
        plt.imshow(imag1_c2[0, :, :, i])
    plt.suptitle('Image 1: Feature maps at conv2')
    plt.savefig('./%s/results/image1_c2.png'%(save_path))

    # image 1, s2
    img1_s2 = activations1[4]

    for i in range(60):
        plt.subplot(6,10,i+1)
        plt.axis('off')
        plt.imshow(img1_s2[0, :, :, i])
    plt.suptitle('Image 1: Feature map at pool2')
    plt.savefig('./%s/results/image1_s2.png'%(save_path))

    # Image 2, c1
    img2_c1 = activations2[1]

    print(img2_c1.shape)
    for i in range(50):
        plt.subplot(5,10,i+1)
        plt.axis('off')
        plt.imshow(img2_c1[0, :, :, i])
    plt.suptitle('Image 2: Feature maps at conv1')
    plt.savefig('./%s/results/image2_c1.png'%(save_path))

    # image 2, s1
    img2_s1 = activations2[2]

    for i in range(50):
        plt.subplot(5,10,i+1)
        plt.axis('off')
        plt.imshow(img2_s1[0, :, :, i])
    plt.suptitle('Image 2: Feature map at pool1')
    plt.savefig('./%s/results/image2_s1.png'%(save_path))

    # image 2, c2
    imag2_c2 = activations2[3]

    for i in range(60):
        plt.subplot(6,10,i+1)
        plt.axis('off')
        plt.imshow(imag2_c2[0, :, :, i])
    plt.suptitle('Image 2: Feature maps at conv2')
    plt.savefig('./%s/results/image2_c2.png'%(save_path))

    # image 2, s2
    img2_s2 = activations1[4]

    for i in range(60):
        plt.subplot(6,10,i+1)
        plt.axis('off')
        plt.imshow(img2_s2[0, :, :, i])
    plt.suptitle('Image 2: Feature map at pool2')
    plt.savefig('./%s/results/image2_s2.png'%(save_path))

if __name__ == '__main__':
    main()
