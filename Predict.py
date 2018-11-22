import tensorflow as tf
import numpy as np
from skimage.io import imread, imsave
from Image import Image
import SuperResolution 
# np.set_printoptions(threshold=np.nan)

batch_size = 30
dataset_size = 1

model_path = './Models/128_32_300_30_100_ssim/'

X_grey = Image.LoadTrainingGreyImage(dataset_size, './Training/X2_grey/')
Y_grey = Image.LoadTrainingGreyImage(dataset_size, './Training/HR_grey/')

X_norm = Image.Normalize(X_grey)
Y_norm = Image.Normalize(Y_grey)

X_cropped = Image.Segment(X_norm, 256)
Y_cropped = Image.Segment(Y_norm, 512)

# print(X_cropped[0])

X_shuffle, Y_shuffle = Image.Shuffle(X_cropped, Y_cropped)

X_final = Image.ExpandDims(X_shuffle)
Y_final = Image.ExpandDims(Y_shuffle)

Image.SaveOutput([Y_final[0]], './y.png')
# print(Y_final[0])
SuperResolution.Test(X_final[0], model_path, './', Y_final[0])