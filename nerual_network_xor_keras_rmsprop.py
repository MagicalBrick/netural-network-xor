import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set a seed, so we can always get same result. It's easier to find problem in your code.
np.random.seed(1)

# Set the train date.
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 1, 1, 0])

# Build a model with one hidden layer.
model = tf.keras.models.Sequential()
# Here is the hidden layer with two units.
# We don't have to add a layer for input date, as the 'input dim' here means two input date size.
model.add(tf.keras.layers.Dense(2, input_dim=2, activation='sigmoid'))
# Output layer
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile the model we built.
# I used another function 'accuracy' to evaluation the training here.
model.compile(loss='binary_crossentropy', metrics=["accuracy"])

# Train the model and show the procession of the training.
hist = model.fit(X, Y, epochs=200, batch_size=4)
print(hist.history.values())

# Show the loss function.
plt.plot(hist.history['loss'])
plt.show()
