import tensorflow as tf
import numpy as np
import imageIO
from skimage.io import imsave

# Hyper params
learning_rate = 0.1
epochs = 50
batch_size = 4
dataset_size = 4
large = 32
small = 16

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

# Load Raw Image
X_train_origin = imageIO.LoadTrainingImages_LR(dataset_size)

print(X_train_origin[0].shape)
print(X_train_origin[1].shape)
print(X_train_origin[2].shape)
print(X_train_origin[3].shape)

Y_train_origin = imageIO.LoadTrainingImages_HR(dataset_size)
X_train_origin = np.reshape(X_train_origin, (np.shape(X_train_origin)[0], np.shape(X_train_origin)[1], np.shape(X_train_origin)[2], 1))
Y_train_origin = np.reshape(Y_train_origin, (np.shape(Y_train_origin)[0], np.shape(Y_train_origin)[1], np.shape(Y_train_origin)[2], 1))

# print(X_train_origin)
# print(X_train_origin[0].shape)
# print(X_train_origin[1].shape)
print(X_train_origin.shape)

# Standardization respect to each color channel per image
X_train_standardized, X_train_mean, X_train_std = imageIO.Standardize(X_train_origin)
Y_train_standardized, Y_train_mean, Y_train_std = imageIO.Standardize(Y_train_origin)

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
bias_conv5 = bias_variable(np.array([batch_size, 1356, 2040, 1]))
Y_predict = tf.nn.relu(deconv2d(conv4, weight_conv5, tf.shape(Y_train)) + bias_conv5)

# Define loss function
loss = tf.reduce_mean(tf.square(tf.subtract(Y_predict, Y_train)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# run
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # for i in range(epochs):
    #     sess.run(optimizer, feed_dict={X_train: X_train_origin[:1], Y_train: Y_train_origin[:1]})
    #     print(loss.eval(feed_dict={X_train: X_train_origin[:1], Y_train: Y_train_origin[:1]}))

    # output = sess.run(Y_predict, feed_dict={X_train: X_train_origin[:1], Y_train: Y_train_origin[:1]})
    # imsave('output.png', output)

    for i in range(epochs):
        sess.run(optimizer, feed_dict={X_train: X_train_standardized[:batch_size], Y_train: Y_train_standardized[:batch_size]})
        print(loss.eval(feed_dict={X_train: X_train_standardized[:batch_size], Y_train: Y_train_standardized[:batch_size]}))

    output = sess.run(Y_predict, feed_dict={X_train: X_train_standardized[:batch_size], Y_train: Y_train_standardized[:batch_size]})
    # output = output_list[0]
    output = np.reshape(output, (np.shape(output)[0], np.shape(output)[1], np.shape(output)[2]))
    imsave('output100_10.png', output[0] * X_train_std[0] + X_train_mean[0])

    # imageIO.InverseStandardize(np.asarray(output))
    # print(output)

sess.close()
