import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

np.random.seed(1)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 1, 1, 0])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(2, input_dim=2, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.5, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"])

hist = model.fit(X, Y, epochs=2000, batch_size=4)
print(hist.history.values())

plt.plot(hist.history['loss'])
plt.show()
