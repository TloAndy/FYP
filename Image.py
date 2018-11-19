import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.color import rgb2grey
from sklearn import preprocessing

class Image:

	@staticmethod
	def Flatten(images):
		images_flatten = []

		for image in images:
			images_flatten.append(np.asarray(image).ravel())

		return images_flatten

	@staticmethod
	def Normalize(images):
		results = []

		for image in images:
			image_norm = image / np.linalg.norm(image)
			# print(np.shape(image_norm))
			image_3dims = image_norm.reshape(np.shape(image_norm)[0], np.shape(image_norm)[1], 1)
			# print(np.shape(image_3dims))
			results.append(image_3dims)

		return results

	@staticmethod
	def SaveGreyScaleImages(size, in_path, out_path):
		names_rgb = [str(x)[1:] + 'x2' + ".png" for x in range(10001, size+10001)]
		images_grey = [imread(in_path + x, as_grey=True) for x in names_rgb]
		names_grey = [out_path + str(x) + '.png' for x in range(1, size+1)]

		for image, name in zip(images_grey, names_grey):
			imsave(name, image)


	@staticmethod
	def LoadTrainingGreyImage(size, in_path):
		return [imread(in_path + str(x) + '.png') for x in range(1, size+1)]

# Image.SaveGreyScaleImages(100, './Training/X2/', './Training/X2_grey/')


