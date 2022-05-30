import tensorflow as tf 
import numpy as np
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense,Flatten


#To get an image from training images -> data.load_data()[0][0][0][10] 
data = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()

#Normalizes the numbers from 0-255 to 0-1
training_images = training_images/255.0
test_images = test_images / 255.0

model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(units=128,activation=tf.nn.relu),
    Dense(units=10,activation=tf.nn.softmax),
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.fit(training_images,training_labels,epochs=5)

model.evaluate(test_images,test_labels)