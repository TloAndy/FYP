import tensorflow as tf
import numpy as np
from Image import Image
import SuperResolution 
# np.set_printoptions(threshold=np.nan)

path_gpu = '/data/ssd/public/kkwong6/Training/'
path_local = './Training/'

# Hyper params
learning_rate = 0.0002
epochs = 500
batch_size = 40
dataset_size = 100

X_grey = Image.LoadTrainingGreyImage(dataset_size, path_gpu + 'X2_grey/')
Y_grey = Image.LoadTrainingGreyImage(dataset_size, path_gpu + 'HR_grey/')

print('finish reading')

X_norm = Image.Normalize(X_grey)
Y_norm = Image.Normalize(Y_grey)

print('finish Normalize')

X_cropped = Image.Segment(X_norm, 256)
Y_cropped = Image.Segment(Y_norm, 512)

print('finish cropping')

X_final = Image.ExpandDims(X_cropped)
Y_final = Image.ExpandDims(Y_cropped)

print('finish ExpandDims')

SuperResolution.Train(X_final, Y_final, Image.ExpandDims(X_norm), learning_rate, epochs, batch_size)










