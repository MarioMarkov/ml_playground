import tensorflow as tf 
import numpy as np
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense,Flatten



model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128,activation=tf.nn.relu),
    Dense(10,activation=tf.nn.softmax),
])

print(model)