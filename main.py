import tensorflow as tf
import numpy as np
from Image import Image
import SuperResolution 
# np.set_printoptions(threshold=np.nan)

# Hyper params
learning_rate = 0.01
epochs = 100
batch_size = 20
dataset_size = 20

X_grey = Image.LoadTrainingGreyImage(dataset_size, './Training/X2_grey/')
Y_grey = Image.LoadTrainingGreyImage(dataset_size, './Training/HR_grey/')

X_norm = Image.Normalize(X_grey)
Y_norm = Image.Normalize(Y_grey)

X_cropped = Image.Segment(X_norm, 256)
Y_cropped = Image.Segment(Y_norm, 512)

X_shuffle, Y_shuffle = Image.Shuffle(X_cropped, Y_cropped)

X_final = Image.ExpandDims(X_shuffle)
Y_final = Image.ExpandDims(Y_shuffle)

SuperResolution.Train(X_final, Y_final, Image.ExpandDims(X_norm), learning_rate, epochs, batch_size)

