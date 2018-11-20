import tensorflow as tf
import numpy as np
from Image import Image
from skimage.io import imsave

# Hyper params
learning_rate = 0.1
epochs = 50
batch_size = 1
dataset_size = 1
large = 32
small = 8
is_grey = True

# Params for each layers
n_filters_conv1 = large
n_channels_conv1 = 1
filter_size_conv1 = 5

n_filters_conv2 = small
n_channels_conv2 = large
filter_size_conv2 = 1

m_mapping_layers = 3
n_filters_conv3 = small
n_channels_conv3 = small
filter_size_conv3 = 3

n_filters_conv4 = large
n_channels_conv4 = small
filter_size_conv4 = 1

n_filters_conv5 = 1
n_channels_conv5 = large
filter_size_conv5 = 9

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(input, filter, stride = [1, 1, 1, 1], padding_type = 'SAME'):
    return tf.nn.conv2d(input, filter, strides=stride, padding=padding_type)

def deconv2d(input, filter, output_shape, stride = [1, 2, 2, 1], padding_type = 'SAME'):
    return tf.nn.conv2d_transpose(input, filter, output_shape=output_shape, strides=stride, padding=padding_type)

def Normalize_1D(images):
    images_flatten = Images.Flatten(images)
    return Image.Normalize(images_flatten)

def Normalize_2D(images):
    return Image.Normalize(images)

X_grey = Image.LoadTrainingGreyImage(dataset_size, './Training/X2_grey/')
Y_grey = Image.LoadTrainingGreyImage(dataset_size, './Training/HR_grey/') 

X_norm = Image.ExpandDims(Normalize_2D(X_grey))
Y_norm = Image.ExpandDims(Normalize_2D(Y_grey))

X_3dims = Image.ExpandDims(X_grey)
Y_3dims = Image.ExpandDims(Y_grey)

imsave('tmp.png', X_grey[0].astype(int))
imsave('tmp1.png', Y_grey[0].astype(int))

X_train = tf.placeholder(tf.float32)
Y_train = tf.placeholder(tf.float32)

# 1st Layer - Features Extraction
print('1st layer')
weight_conv1 = weight_variable([filter_size_conv1, filter_size_conv1, n_channels_conv1, n_filters_conv1])
bias_conv1 = bias_variable([n_filters_conv1])
conv1 = tf.nn.relu(conv2d(X_train, weight_conv1) + bias_conv1)

#  2nd - Shrinking
print('2nd layer')
weight_conv2 = weight_variable([filter_size_conv2, filter_size_conv2, n_channels_conv2, n_filters_conv2])
bias_conv2 = bias_variable([n_filters_conv2])
conv2 = tf.nn.relu(conv2d(conv1, weight_conv2) + bias_conv2)

# 3nd Layer - Non-linear mapping
print('3rd layer')
conv3_input = []
conv3_input.append(conv2)

for i in range(m_mapping_layers):
    weight_conv3 = weight_variable([filter_size_conv3, filter_size_conv3, n_channels_conv3, n_filters_conv3])
    bias_conv3 = bias_variable([n_filters_conv3])
    conv3 = tf.nn.relu(conv2d(conv3_input[i], weight_conv3) + bias_conv3)
    conv3_input.append(conv3)

# 4th Layer - Expanding
print('4th layer')
weight_conv4 = weight_variable([filter_size_conv4, filter_size_conv4, n_channels_conv4, n_filters_conv4])
bias_conv4 = bias_variable([n_filters_conv4])
conv4 = tf.nn.relu(conv2d(conv3, weight_conv4) + bias_conv4)

# 5th Layer - Deconvolution
print('5th layer')
weight_conv5 = weight_variable([filter_size_conv5, filter_size_conv5, n_filters_conv5, n_channels_conv5])
bias_conv5 = bias_variable(np.array([batch_size, 1404, 2040, 1]))
Y_predict = tf.nn.relu(deconv2d(conv4, weight_conv5, tf.shape(Y_train)) + bias_conv5)

# Define loss function
loss = tf.reduce_mean(tf.square(tf.subtract(Y_predict, Y_train)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# run
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(epochs):
        sess.run(optimizer, feed_dict={X_train: X_norm, Y_train: Y_norm})
        print(loss.eval(feed_dict={X_train: X_norm, Y_train: Y_norm}))

    output = sess.run(Y_predict, feed_dict={X_train: X_3dims, Y_train: Y_3dims})
    output = np.reshape(output, (np.shape(output)[1], np.shape(output)[2]))
    # output = output * std + mean
    print(output)
    im = output.astype(int)
    # print(np.shape(output))
    # output = np.reshape(output, (np.shape(output)[1], np.shape(output)[2], np.shape(output)[3]))
    imsave('output100_10.png', im)

sess.close()
