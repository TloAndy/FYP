import tensorflow as tf
import numpy as np
from Image import Image

# Hyper params
learning_rate = 0.1
epochs = 1
batch_size = 1
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

def deconv2d(input, filter, output_shape, stride = [1, 4, 4, 1], padding_type = 'SAME'):
    return tf.nn.conv2d_transpose(input, filter, output_shape=output_shape, strides=stride, padding=padding_type)

# Functions for each layer
def FeatureExtraction(X_train):
    print('1st layer')
    weight_conv1 = weight_variable([filter_size_conv1, filter_size_conv1, n_channels_conv1, n_filters_conv1])
    bias_conv1 = bias_variable([n_filters_conv1])
    conv1 = tf.nn.relu(conv2d(X_train, weight_conv1) + bias_conv1)
    return conv1


def Shrinking(conv1):
    print('2nd layer')
    weight_conv2 = weight_variable([filter_size_conv2, filter_size_conv2, n_channels_conv2, n_filters_conv2])
    bias_conv2 = bias_variable([n_filters_conv2])
    conv2 = tf.nn.relu(conv2d(conv1, weight_conv2) + bias_conv2)
    return conv2


def NonLinearMapping(conv2):
    print('3rd layer')
    conv3_input = []
    conv3_input.append(conv2)

    for i in range(m_mapping_layers):
        weight_conv3 = weight_variable([filter_size_conv3, filter_size_conv3, n_channels_conv3, n_filters_conv3])
        bias_conv3 = bias_variable([n_filters_conv3])
        conv3 = tf.nn.relu(conv2d(conv3_input[i], weight_conv3) + bias_conv3)
        conv3_input.append(conv3)

    return conv3

def Expanding(conv3):
    print('4th layer')
    weight_conv4 = weight_variable([filter_size_conv4, filter_size_conv4, n_channels_conv4, n_filters_conv4])
    bias_conv4 = bias_variable([n_filters_conv4])
    conv4 = tf.nn.relu(conv2d(conv3, weight_conv4) + bias_conv4)
    return conv4

def Deconvolution(conv4, Y_train, size_y):
    print('5th layer')
    weight_conv5 = weight_variable([filter_size_conv5, filter_size_conv5, n_filters_conv5, n_channels_conv5])
    bias_conv5 = bias_variable(np.array([batch_size, 1, size_y, 1]))
    # bias_conv5 = bias_variable(np.asarray(Y_train.get_shape().as_list()))
    Y_predict = tf.nn.relu(deconv2d(conv4, weight_conv5, tf.shape(Y_train)) + bias_conv5)
    return Y_predict

def Training(X_images, Y_images, is_print_loss):

    # size_x = np.shape(X_images)[0]
    size_y = np.shape(Y_images)[0]
    # tmp = np.shape(Y_images)[1]

    X_train = tf.placeholder(tf.float32)
    Y_train = tf.placeholder(tf.float32)

    conv1 = FeatureExtraction(X_train)
    conv2 = Shrinking(conv1)
    conv3 = NonLinearMapping(conv2)
    conv4 = Expanding(conv3)
    Y_predict = Deconvolution(conv4, Y_train, size_y)

    # Define loss function and optimizer
    loss = tf.reduce_mean(tf.square(tf.subtract(Y_predict, Y_train)))
    # loss = tf.image.ssim(Y_predict, Y_train, max_val=1.0)
    # loss = tf.image.ssim_multiscale(Y_predict, Y_train, max_val=1.0)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # run
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    sess.run(optimizer, feed_dict={X_train: X_images, Y_train: Y_images})

    if is_print_loss:
        print(loss.eval(feed_dict={X_train: X_images, Y_train: Y_images}))

    sess.close()


X_grey = Image.LoadTrainingGreyImage(2, './Training/X2_grey/')
Y_grey = Image.LoadTrainingGreyImage(2, './Training/HR_grey/')

X_grey_flatten = Image.Flatten(X_grey)
Y_grey_flatten = Image.Flatten(Y_grey)

X_final = Image.Normalize(X_grey_flatten)
Y_final = Image.Normalize(Y_grey_flatten)


for i in range(epochs):

    for j in range(len(X_final)):
        print(X_final[j].shape)
        print(Y_final[j].shape)
        Training([X_final[j]], [Y_final[j]], True)
        break












