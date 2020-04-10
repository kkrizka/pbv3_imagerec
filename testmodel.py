#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

#
# Load and label the dataset
CLASS_NAMES=np.arange(0,10)

def get_label(file_path):
    file_name=tf.strings.split(file_path, os.path.sep)[-1]
    digit=tf.strings.substr(file_name, 0, 1)
    return tf.strings.to_number(digit)

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, (100, 100))
    img = tf.image.convert_image_dtype(img, tf.float32)

    return img, label

data=tf.data.Dataset.list_files('testimages/*png')
print(data)

for f in data.take(5):
    print(f.numpy())

labeled_data = data.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

for image,label in labeled_data.take(10):
    print(image.numpy().shape, label.numpy())

#
# Prepare data for training
def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n][:,:,0])
        plt.title(label_batch[n])
        plt.axis('off')
    plt.show()

train_ds = labeled_data.batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

image_batch, label_batch = next(iter(train_ds))
show_batch(image_batch.numpy(), label_batch.numpy())

#
# Make model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100,100,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
    ])
model.summary()

model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

model.fit(train_ds, epochs=5)
model.evaluate(train_ds, verbose=2)

image_batch, label_batch = next(iter(train_ds))
print(model(image_batch).numpy())

predict_batch=model.predict(image_batch)
print(predict_batch)
print(np.argmax(predict_batch))
predict_batch=np.argmax(model.predict(image_batch),axis=1)
print(predict_batch)
show_batch(image_batch, predict_batch)
