import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def training_data(train_dir, batch_size, img_height, img_width, colour_option):
  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_dir,
  color_mode=colour_option,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
  return train_ds

def validation_data(val_dir, batch_size, img_height, img_width, colour_option):
  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  val_dir,
  #RGB for colour images grayscale for b&w
  color_mode=colour_option,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
  return val_ds


def model_build(img_height, img_width, colour_chanels, hl1, hl2, hl3, hl4, hl5, hl6):
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

    model = models.Sequential()
    model.add(layers.Conv2D(512, (3, 3), activation='swish', input_shape=(img_height, img_width, colour_chanels)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(1024, (3, 3), activation='swish'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(2048, (3, 3), activation='swish'))
    model.add(layers.Flatten())
    model.add(layers.Dense(hl1, activation='swish'))
    model.add(layers.Dense(hl2, activation='swish'))
    model.add(layers.Dense(hl3, activation='swish'))
    model.add(layers.Dense(hl4, activation='swish'))
    model.add(layers.Dense(hl5, activation='swish'))
    model.add(layers.Dense(hl6, activation='swish'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
    return model

strategy = tf.distribute.MirroredStrategy()
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
time_start = tf.timestamp()

batch_size = 250
num_classes = 10
num_epochs = 200
hl1 = 1024
hl2 = 1024
hl3 = 1024
hl4 = 1024
hl5 = 1024
hl6 = 1024
img_height = 64
img_width = 64
colour_chanels = 3
#RGB for colour images grayscale for b&w
colour_option = 'RGB'
#title = "Digits "
#train_dir = "/home/ks222/chrisknight/PythonCNN/Datasets/Digits/train"
#val_dir = "/home/ks222/chrisknight/PythonCNN/Datasets/Digits/test"
#save_dir = "/home/ks222/chrisknight/PythonCNN/ModelSave/Digits"
#title = "Fashion "
#train_dir = "/home/ks222/chrisknight/PythonCNN/Datasets/Fashion/train"
#val_dir = "/home/ks222/chrisknight/PythonCNN/Datasets/Fashion/test"
#save_dir = "/home/ks222/chrisknight/PythonCNN/ModelSave/Fashion"
title = "EuroSat "
train_dir = "/home/ks222/chrisknight/PythonCNN/Datasets/EuroSat/train"
val_dir = "/home/ks222/chrisknight/PythonCNN/Datasets/EuroSat/test"
save_dir = "/home/ks222/chrisknight/PythonCNN/ModelSave/EuroSat"


multi_training = training_data(train_dir, batch_size, img_height, img_width, colour_option)
multi_validation = validation_data(val_dir, batch_size, img_height, img_width, colour_option)

AUTOTUNE = tf.data.AUTOTUNE

multi_training = multi_training.cache().prefetch(buffer_size=AUTOTUNE)
multi_validation = multi_validation.cache().prefetch(buffer_size=AUTOTUNE)
with strategy.scope():
    multi_model = model_build(img_height, img_width, colour_chanels, hl1, hl2, hl3, hl4, hl5, hl6)
    history = multi_model.fit(
        multi_training,
        validation_data=multi_validation,
        epochs=num_epochs,
        verbose=1
        )
time_end = tf.timestamp()
evaluation = str(multi_model.evaluate(multi_validation,verbose = 2))
last_chars = evaluation[ 20: 27: 1]
print(last_chars)
total_time = "Total Time%8.2f Seconds" % float(time_end - time_start)
print(total_time)
hl1_str = str(hl1)
hl2_str = str(hl2)
hl3_str = str(hl3)
hl4_str = str(hl4)
hl5_str = str(hl5)
hl6_str = str(hl6)
file_extention = ".png"
filename = title + "Hidden layers " + "HL1 " + hl1_str + " HL2 " + hl2_str + " HL3 " + hl3_str + " HL4 " + hl4_str + " HL5 " + hl5_str + " HL6 " + hl6_str + " " + total_time + " " + last_chars + file_extention
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.2, 1])
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(filename)
multi_model.save(save_dir)
