# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 08:54:45 2023

@author: Asus
"""
#%%
"""
In this Code, i want to build a pipeline to classify brain tumor images
if the tumors are hemorrage or non

"""
#%%
# here i will import needed libraries
# Keras is needed preprocessing image data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Sequential to build the CNN model
from tensorflow.keras.models import Sequential
# and these functions to work on the layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# Adam to edit the learning rate
from tensorflow.keras.optimizers import Adam
# Classification report and confusion matrix to further analyze the results
from sklearn.metrics import classification_report, confusion_matrix
#%%
# Here i will assign variables for the images in their directories
train_dir = "F:\\Brain Tumor Images Dataset\\training_set"
validation_dir = "F:\\Brain Tumor Images Dataset\\validation_set"
test_dir = "F:\\Brain Tumor Images Dataset\\test_set"
#%%
# setting the images height and width
img_width, img_height = 100, 100
#%%
# Create data generators with data augmentation for training
# the rescale is for the pixels
# rotation range for adding variability for the prientation of images
# width and height shift is to randomly shift the images
# ZCA Whitening to decorrelate the features of the images, it can enhance the performance of a neural network
# zca epsilon is A small constant added to the denominator for numerical stability for the performance of ZCA Whitening
# Shearing involves shifting one part of an image, in a fixed direction, by a specified fraction
# zoom range introduces variability in the scale of objects in the images.
# vertical flip is to introduces variability in the scale of objects in the images.
# fill mode is a Strategy for filling in newly created pixels after a rotation or a width/height shift
train_datagen = ImageDataGenerator(
    rescale=20./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zca_whitening=True,
    zca_epsilon=1e-06,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)
# Validation and test data should not be augmented
validation_datagen = ImageDataGenerator(rescale=20./255)
test_datagen = ImageDataGenerator(rescale=20./255)
#%%
# Here i will the flow_from_directory method to generate and normalize the image data
# and hee i can change batch size, images size and class modes
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_width, img_height), batch_size=64, class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(img_width, img_height), batch_size=64, class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(img_width, img_height), batch_size=64, class_mode='binary')
#%%
# Build the CNN model
# input shape have 3 channels with the images size
# strides is for the step Size of the convolutional kernel or filter
# size and step size of the pooling window
# Pooling is a down-sampling operation that reduces the spatial dimensions of the input volume
# flatten the input to 1d array
model = Sequential([
    Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=(img_width, img_height, 3),strides=(2, 2), padding='valid'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='valid'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='valid'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Use 'sigmoid' for binary classification
])
#%%
# Adam is an optimization algorithm used to adapts the learning rates of each parameter individually by computing adaptive learning rates for them.
# learning rate determines the step size at each iteration during optimization
custom_optimizer = Adam(learning_rate=0.006)
# compile Configures the model for training
model.compile(optimizer=custom_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#%%
# fitting the model on the validaiton set
model.fit(train_generator, epochs=20, validation_data=validation_generator)
#%%
# Test the model ontest set
test_pred = model.predict(test_generator)

# creating a binary representation of predictions and 0.5 will be a threshold
test_pred_binary = (test_pred > 0.5).astype(int)
print("Confusion Matrix:")
print(confusion_matrix(test_generator.classes, test_pred_binary))
print("Classification Report:")
print(classification_report(test_generator.classes, test_pred_binary))
#%%
"""
In this code, i used some preprocess methods and then applied the CNN model.
I got the results of accuracy, f1, precision and recall.
Augementation, learning rate, rescale and images width and height seems to b  ethe most effect hyperparameters here

"""
#%%
"""
Dataset Harvard Refrencing:
Simeon. A, (2019) Brain Tumor Images Dataset, Kaggle. Available at: https://www.kaggle.com/datasets/simeondee/brain-tumor-images-dataset/data (Accessed: July 25, 2023). 
    """