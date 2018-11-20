import tensorflow as tf
import numpy as np
from Image import Image
import SuperResolution 

# Hyper params
learning_rate = 0.5
epochs = 100
batch_size = 1
dataset_size = 1

X_grey = Image.LoadTrainingGreyImage(dataset_size, './Training/X2_grey/')
Y_grey = Image.LoadTrainingGreyImage(dataset_size, './Training/HR_grey/')

X_norm = Image.Normalize(X_grey)
Y_norm = Image.Normalize(Y_grey)

X_final = Image.ExpandDims(X_norm)
Y_final = Image.ExpandDims(Y_norm)

SuperResolution.Train(X_final, Y_final, learning_rate, epochs)

