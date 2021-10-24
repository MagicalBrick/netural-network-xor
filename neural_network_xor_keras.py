import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf

np.random.seed(1)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 1, 1, 0])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(2, input_dim=2, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

opt = keras.optimizers.Adam(learning_rate=0.5)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

hist = model.fit(X, Y, epochs=300, batch_size=4)
print(hist.history.values())

plt.plot(hist.history['loss'])
plt.show()
