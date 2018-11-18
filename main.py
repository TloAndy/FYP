import tensorflow as tf
import numpy as np
from Image import Image
import SuperResolution 

# Hyper params
learning_rate = 0.1
epochs = 1
batch_size = 1
dataset_size = 4

X_grey = Image.LoadTrainingGreyImage(2, './Training/X2_grey/')
Y_grey = Image.LoadTrainingGreyImage(2, './Training/HR_grey/')

X_grey_flatten = Image.Flatten(X_grey)
Y_grey_flatten = Image.Flatten(Y_grey)

X_final = Image.Normalize(X_grey_flatten)
Y_final = Image.Normalize(Y_grey_flatten)


for i in range(epochs):

    for j in range(len(X_final)):
        SuperResolution.Training([X_final[j]], [Y_final[j]], True, learning_rate)
        break