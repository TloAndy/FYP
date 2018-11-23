import tensorflow as tf
import numpy as np
from Image import Image
import SuperResolution 
# np.set_printoptions(threshold=np.nan)

path_gpu = '/data/ssd/public/kkwong6/Training/'
path_local = './Training/'

# Hyper params
learning_rate = 0.0002
epochs = 1
batch_size = 40
dataset_size = 200

X_grey = Image.LoadTrainingGreyImage(dataset_size, path_local + 'X2_grey/')
Y_grey = Image.LoadTrainingGreyImage(dataset_size, path_local + 'HR_grey/')

X_norm = Image.Normalize(X_grey)
Y_norm = Image.Normalize(Y_grey)

X_cropped = Image.Segment(X_norm, 256)
Y_cropped = Image.Segment(Y_norm, 512)

X_final = Image.ExpandDims(X_cropped)
Y_final = Image.ExpandDims(Y_cropped)

SuperResolution.Train(X_final, Y_final, Image.ExpandDims(X_norm), learning_rate, epochs, batch_size)










