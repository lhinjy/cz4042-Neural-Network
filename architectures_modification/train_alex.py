import os
import numpy as np
import random 

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from prep import data
from alexnet import modification_2

# Hyper parameters 
dropout_prob = 0.5
weight_decay = 0.001
l_r = 0.001
batch_size = 32
epochs = 100
momentum = 0.9
steps_per_epoch = 50
validation_steps = 30

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)


save_path = ('alexnet')
if not os.path.isdir(save_path):
    os.makedirs(save_path)


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


def main():
  train_df,test_df = data()
  model = modification_2()

  optimizer = keras.optimizers.SGD(learning_rate=l_r, momentum=momentum)
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


  model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
  history = model.fit(
      train_df,
      batch_size=batch_size,
      epochs=epochs,
      steps_per_epoch=steps_per_epoch,
      validation_data=test_df,
      validation_steps=validation_steps )
    

  train_loss = history.history['loss']
  val_loss = history.history['val_loss']
  train_acc = history.history['accuracy']
  test_acc = history.history['val_accuracy']

  model.save('./%s/alexnet_mod2.h5'%(save_path))


  # Save the plot for losses
  plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train')
  plt.plot(range(1, len(val_loss) + 1), val_loss, label='Test')
  plt.title('Model Loss: AlexNet mod 2')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(loc='best')
  plt.savefig('./%s/mod2_loss.png'%(save_path))
  plt.show()
  plt.close()

  # Save the plot for accuracies
  plt.plot(range(1, len(train_acc) + 1), train_acc, label='Train')
  plt.plot(range(1, len(test_acc) + 1), test_acc, label='Test')
  plt.title('Model Accuracy: AlexNet mod 2')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(loc = 'best')
  plt.savefig('./%s/mod2_accuracy.png'%(save_path))
  plt.show()
  plt.close()

if __name__ == '__main__':
    main()