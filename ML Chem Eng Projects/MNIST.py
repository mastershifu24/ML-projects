#Title: Building a Digit Classifier using the MNIST Dataset - A Tutorial

#In this tutorial, we will walk you through the process of training a machine learning model to recognize handwritten digits
#from 0 to 9. By the end of this tutorial, you will have a working digit classifier that can predict the correct digit 
#given an input image.

#Step 1: Dataset Overview:
#We begin by providing an overview of the MNIST dataset. It consists of 60,000 training images and 10,000 
#test images, each of which is a grayscale image of size 28x28 pixels. The dataset is widely used as a 
#benchmark for image classification tasks.

#Step 2: Data Preparation:
#Steps involved in preparing the data for training the model includes loading the dataset using various libraries, 
#normalizing the pixel values to a range between 0 and 1, and splitting the dataset into training and test sets. 

#Step 3: Model Architecture:
#In this tutorial, we will use a simple feedforward neural network with two dense layers.
#The input layer takes the flattened 28x28 image pixels as input, followed by a hidden layer with ReLU activation, 
#and an output layer with softmax activation to produce class probabilities.

#Step 4: Model Compilation and Training:
The model will be trained using the training set, with the validation set used to monitor the model's performance and 
#prevent overfitting. Specify the number of epochs and batch size for training and discuss their significance.

#Step 5: Model Evaluation:
#After training the model, evaluate its performance on the test set. 
#Calculate and display the test loss and test accuracy, which provide insights into how well the 
#model generalizes to unseen data.

#Step 6: Prediction and Comparison:
#Use the trained model to make predictions on a subset of the test set. We then display the predicted labels alongside the 
#true labels to compare the model's predictions with the ground truth. 

#Step 7: Visualization:
#We will visualize the training and validation loss/accuracy over epochs using line plots. 
#These visualizations help in understanding the model's training progress, identifying overfitting, and 
#analyzing its performance.

#Conclusion:
#We have successfully built a digit classifier using the MNIST dataset, and here, I've explained the process step-by-step. 
#Experiment with different model architectures or datasets to further your understanding.

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Loading the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizing pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Splitting the data into training, validation, and testing sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Reshaping the data for a CNN (if using)
# X_train = X_train.reshape(-1, 28, 28, 1)
# X_val = X_val.reshape(-1, 28, 28, 1)
# X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encoding the labels
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# Building the model architecture
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # Input layer
model.add(Dense(128, activation='relu'))  # Hidden layer
model.add(Dense(10, activation='softmax'))  # Output layer

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

# Evaluating the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Making predictions on new, unseen images
predictions = model.predict(X_test[:5])  # Selecting the first 5 test images
predicted_labels = np.argmax(predictions, axis=1)

# Comparing with true labels
true_labels = np.argmax(y_test[:5], axis=1)
for i in range(5):
    print(f"Predicted: {predicted_labels[i]}, True: {true_labels[i]}")

# Visualizing the training and validation loss/accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Saving and loading the model (optional)
# model.save('digit_classifier.h5')
# loaded_model = tf.keras.models.load_model('digit_classifier.h5')
