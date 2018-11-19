import tensorflow as tf
import numpy as np
from Image import Image
import SuperResolution 

# Hyper params
learning_rate = 0.1
epochs = 1
batch_size = 1
dataset_size = 1

def Normalize_1D(images):
    images_flatten = Images.Flatten(images)
    return Image.Normalize(images_flatten)

def Normalize_2D(images):
    return Image.Normalize(images)

X_grey = Image.LoadTrainingGreyImage(dataset_size, './Training/X2_grey/')
Y_grey = Image.LoadTrainingGreyImage(dataset_size, './Training/HR_grey/')

X_norm = Normalize_2D(X_grey)
Y_norm = Normalize_2D(Y_grey)

# print(np.shape(X_norm))
# print(np.shape(Y_norm))

# X_norn = Normalize_1D(X_grey)
# Y_norm = Normalize_1D(Y_grey)

for i in range(epochs):

    for j in range(dataset_size):
        SuperResolution.Training([X_norm[j]], [Y_norm[j]], learning_rate)