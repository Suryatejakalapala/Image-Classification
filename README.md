# Image Classification with CNN Sequential API using a Pipeline

## Getting Started

To get started, make sure you have the following dependencies installed:

- Python 3.6+
- TensorFlow 2.x
- Keras 2.x
- NumPy
- Matplotlib

Once you have installed these dependencies, you can clone this repository and begin working on the project.

## The Dataset

The dataset used in this project is a randomly generated dataset, not a specific named dataset. It has been scraped from a publicly available source.

## The Model

The model used in this project is a Convolutional Neural Network (CNN) with the following architecture:

```python
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

The model is trained using the Adam optimizer and the categorical cross-entropy loss function.

## The Pipline
The Pipline is used to automate the process of training and evaluating the model. The Pipline takes care of loading the dataset, preprocessing the images, training the model, and evaluating the model.
