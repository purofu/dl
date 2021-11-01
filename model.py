import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Rescaling, RandomFlip, RandomRotation, RandomZoom,
									Dense, Flatten, Dropout, Conv2D, MaxPooling2D)
from tensorflow.keras.utils import image_dataset_from_directory, plot_model
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.applications import VGG16

# Basic parameters
batch_size = 32
img_height = 150
img_width = 150

# Datasets directories
train_path = 'data_set/seg_train/seg_train/'
test_path = 'data_set/seg_test/seg_test/'
pred_path = 'data_set/seg_pred/seg_pred/'

# Load data
def load_data(path, labels):
	dataset = image_dataset_from_directory(
		directory=path,
		labels=labels,
		seed=123,
		image_size=(img_height, img_width),
		batch_size=batch_size
	)
	return dataset
	
# Datasets
train_ds = load_data(train_path, labels='inferred')
test_ds = load_data(test_path, labels='inferred')
pred_ds = load_data(pred_path, labels=None)

# Image labels
class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

#Translate datasets to trainable data
scaling = Rescaling(1. / 255)

train_ds = train_ds.map(lambda x, y: (scaling(x), y))
test_ds = test_ds.map(lambda x, y: (scaling(x), y))
pred_ds = pred_ds.map(lambda x: scaling(x))

model = Sequential([
	Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
	MaxPooling2D(),
	Conv2D(32, 3, padding='same', activation='relu'),
	MaxPooling2D(),
	Conv2D(64, 3, padding='same', activation='relu'),
	MaxPooling2D(),
	Flatten(),
	Dense(128, activation='relu'),
	Dense(6, activation='softmax')
])

model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])
  
model.summary()

epochs = 10

history = model.fit(train_ds, validation_data=test_ds, epochs=epochs)


