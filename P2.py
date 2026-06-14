# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 21:13:56 2026

@author: Saranya
"""
import tensorflow



import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(x_train.shape)
print(y_train.shape)

import matplotlib.pyplot as plt

#This displays a handwritten digit along with its correct label.

plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()



import tensorflow as tf

# import own dataset

train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=(224,224),
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=(224,224),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=42
)