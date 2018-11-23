import numpy as np
import tensorflow as tf
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.color import rgb2grey
from sklearn import preprocessing
from random import shuffle
# np.set_printoptions(threshold=np.nan)

class Image:

	@staticmethod
	def SegmentImage(images):
		segmentSize = 60
		segments = []
		for index, image in enumerate(images):
			height, width = np.shape(image)[0], np.shape(image)[1]
			print('image', index, ': Height(', height, ') Width(', width, ')')
			XSegmentNum = width//segmentSize
			YSegmentNum = height//segmentSize
			for x in range(XSegmentNum):
				for y in range(YSegmentNum):
					segments.append(image[y*segmentSize:(y+1)*segmentSize, x*segmentSize:(x+1)*segmentSize])
			print('Cropped into', XSegmentNum*YSegmentNum, 'images')

		return segments

	@staticmethod
	def Flatten(images):
		images_flatten = []

		for image in images:
			images_flatten.append(np.asarray(image).ravel())

		return images_flatten

	@staticmethod
	def Recover(images):
		return [((image + 1) * 255.0 / 2.0).astype(int) for image in images]

	@staticmethod
	def Normalize(images):
		return [((image * 2) / 255.0) - 1 for image in images]

	@staticmethod
	def ExpandDims(images):
		return [np.reshape(image, (np.shape(image)[0], np.shape(image)[1], 1)) for image in images]

	@staticmethod
	def Shuffle(X_images, Y_images):
		images_count = np.shape(X_images)[0]
		shuffled_index = list(range(images_count))
		shuffle(shuffled_index)
		return [X_images[i] for i in shuffled_index], [Y_images[i] for i in shuffled_index]

	@staticmethod
	def Segment(images, size):
		segmentSize = size
		segments = []

		for index, image in enumerate(images):

			height, width = np.shape(image)[0], np.shape(image)[1]
			XSegmentNum = width//segmentSize
			YSegmentNum = height//segmentSize

			for x in range(XSegmentNum):
				for y in range(YSegmentNum):
					segments.append(image[y*segmentSize:(y+1)*segmentSize, x*segmentSize:(x+1)*segmentSize])

		return segments

	@staticmethod
	def SaveGreyScaleImages(size, in_path, out_path):
		names_rgb = [str(x)[1:] + 'x2' +  ".png" for x in range(10101, size+10101)]
		images_grey = [imread(in_path + x, as_grey=True) for x in names_rgb]
		names_grey = [out_path + str(x) + '.png' for x in range(101, size+101)]

		for image, name in zip(images_grey, names_grey):
			imsave(name, image)


	@staticmethod
	def LoadTrainingGreyImage(size, in_path):
		return [(imread(in_path + str(x) + '.png') / 255).astype(int) for x in range(1, size+1)]

	# assume output has 4 dims (model output NHWC and N == 1)
	@staticmethod
	def SaveOutput(output, path):

		image = np.reshape(output, (np.shape(output)[1], np.shape(output)[2]))
		print(image)
		# imsave(path, image)

		image = ((image + 1) * 255.0 / 2.0)
		# print(image.dtype)
		image_int = image.astype(int)
		# print(image_int)

		imsave(path, image_int)


# Image.SaveGreyScaleImages(100, './Training/X2/', './Training/X2_grey/')




