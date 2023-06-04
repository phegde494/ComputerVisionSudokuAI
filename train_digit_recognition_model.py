import tensorflow as tf
from tensorflow import keras

# Load the MNIST data set to train our digit recognition model
# This dataset has thousands of 28x28 grayscale digit images
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel vals to doubles in range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define our digit recognition model to take in 28x28 pixel images
# ReLU function optimizes hidden layer while Softmax optimizes output layer (good for multi-class classification)
digit_recognition_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Train model 
digit_recognition_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 15 epochs is near ideal for MNIST dataset
digit_recognition_model.fit(x_train, y_train, epochs=15)

test_loss, test_accuracy = digit_recognition_model.evaluate(x_test, y_test)

digit_recognition_model.save("trained_digit_model.h5")

print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)
print(digit_recognition_model.input_shape)