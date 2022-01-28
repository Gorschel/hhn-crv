#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import pathlib

from sklearn.model_selection import train_test_split

IMAGE_SHAPE = (224, 224)

data_dir = pathlib.Path(pathlib.Path.home(), 'PycharmProjects/hhn-crv/data')

data_stamped = list(data_dir.glob('stamped_cropped/*'))

data_unstamped = list(data_dir.glob('unstamped_cropped/*'))

letters_images_dict = {
  'stamped': list(data_dir.glob('stamped_cropped/*')),
  'unstamped': list(data_dir.glob('unstamped_cropped/*')),
}

letters_labels_dict = {
  'stamped': 0,
  'unstamped': 1,
}

X, y = [], []

for category, images in letters_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img, IMAGE_SHAPE)
        X.append(resized_img)
        y.append(letters_labels_dict[category])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25)

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
pretrained_model_without_top_layer = hub.KerasLayer(feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

num_classes = 2

model = tf.keras.Sequential([pretrained_model_without_top_layer, tf.keras.layers.Dense(num_classes)])

model.summary()

model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['acc'])

model.fit(X_train_scaled, y_train, epochs=5)
print("eval")

model.evaluate(X_test_scaled, y_test)
print(len(y_test))

result = model.predict(np.array([X_test_scaled[0]]))

predicted_label_index = np.argmax(result)
print(predicted_label_index)
failcount = 0
for i in range(len(X_test)):
    result = model.predict(np.array([X_test_scaled[i]]))
    predicted_label_index = np.argmax(result)
    if predicted_label_index != y_test[i]:
        print(i)
        failcount += 1
        plt.imshow(X[i])
        plt.show()

print("score:")
print(1-(failcount/len(y_test)))

