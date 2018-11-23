import tensorflow as tf
import numpy as np
from Image import Image
from skimage.io import imread, imsave
# np.set_printoptions(threshold=np.nan)

# Params for each layers
large = 64
small = 32

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

def prelu(_x, num, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"+num):
        _alpha = tf.get_variable("prelu"+num, shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

# Functions for each layer
def FeatureExtraction(X_train):
    weight_conv1 = weight_variable([filter_size_conv1, filter_size_conv1, n_channels_conv1, n_filters_conv1])
    bias_conv1 = bias_variable([n_filters_conv1])
    conv1 = prelu(conv2d(X_train, weight_conv1) + bias_conv1, '1')
    return conv1


def Shrinking(conv1):
    weight_conv2 = weight_variable([filter_size_conv2, filter_size_conv2, n_channels_conv2, n_filters_conv2])
    bias_conv2 = bias_variable([n_filters_conv2])
    conv2 = prelu(conv2d(conv1, weight_conv2) + bias_conv2, '2')
    return conv2


def NonLinearMapping(conv2):
    conv3_input = []
    conv3_input.append(conv2)

    for i in range(m_mapping_layers):
        weight_conv3 = weight_variable([filter_size_conv3, filter_size_conv3, n_channels_conv3, n_filters_conv3])
        bias_conv3 = bias_variable([n_filters_conv3])
        conv3 = prelu(conv2d(conv3_input[i], weight_conv3) + bias_conv3, str(3 + i))
        conv3_input.append(conv3)

    return conv3

def Expanding(conv3):
    weight_conv4 = weight_variable([filter_size_conv4, filter_size_conv4, n_channels_conv4, n_filters_conv4])
    bias_conv4 = bias_variable([n_filters_conv4])
    conv4 = prelu(conv2d(conv3, weight_conv4) + bias_conv4, '6')
    return conv4

def Deconvolution(conv4, Y_train, batch_size, out_h, out_w):
    weight_conv5 = weight_variable([filter_size_conv5, filter_size_conv5, n_filters_conv5, n_channels_conv5])
    bias_conv5 = bias_variable(np.array([batch_size, out_h, out_w, 1]))
    Y_predict = prelu(deconv2d(conv4, weight_conv5, tf.shape(Y_train)) + bias_conv5, '7')
    return Y_predict

def Train(X_images, Y_images, test_images, learning_rate, epochs, batch_size):

    current_batch_size = 1
    X_train = tf.placeholder(tf.float32, name='X')
    Y_train = tf.placeholder(tf.float32, name='Y')
    output_h = np.shape(Y_images)[1]
    output_w = np.shape(Y_images)[2]

    conv1 = FeatureExtraction(X_train)
    conv2 = Shrinking(conv1)
    conv3 = NonLinearMapping(conv2)
    conv4 = Expanding(conv3)
    Y_predict = Deconvolution(conv4, Y_train, current_batch_size)
    print(Y_predict)

    # Define loss function and optimizer
    loss = tf.image.ssim(Y_predict, Y_train, max_val=2.0)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(-loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.990

    with tf.Session(config=config) as sess:
        print('hello world')
        sess.run(init)
        start_index = 0
        end_index = batch_size
        total = np.shape(X_images)[0]
        feed_all = False
        iter_count = 0

        for epoch in range(epochs):

            X_images, Y_images = Image.Shuffle(X_images, Y_images)

            while True:

                X_batch, Y_batch = X_images[start_index:end_index], Y_images[start_index:end_index]
                current_batch_size = end_index - start_index
                sess.run(train_step, feed_dict={X_train:X_batch, Y_train:Y_batch})
                start_index += batch_size
                end_index += batch_size

                if epoch % 20 == 0:
                    loss_value = np,mean(loss.eval(feed_dict={X_train: X_batch, Y_train: Y_batch}))
                    print('In epoch: ' + str(epoch) + ' iteration = ' + str(iter_count) + ', loss = ' + str(loss_value))

                if end_index >= total:
                    end_index = total
                
                if start_index >= total:
                    start_index = 0
                    end_index = batch_size
                    break

                iter_count += 1

            if epoch % 200 == 0 and epoch > 1:
                saver.save(sess, './model-' + str(epoch) + '.ckpt')

        saver.save(sess, './model-final' + '.ckpt')

    sess.close()

def Test(X, model_path, output_path):

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path + 'model.ckpt.meta')
        saver.restore(sess, model_path + 'model.ckpt')

        Y = np.zeros(shape=(1, np.shape(input)[0] * 2, np.shape(input)[1] * 2, np.shape(input)[2]))
        output = sess.run('Relu_6:0', feed_dict={'X': [input], 'Y': [Y]})
        output = np.reshape(output, (np.shape(output)[1], np.shape(output)[2]))
        output = np.clip(output, -1, 1)
        output *= -1
        image = ((output + 1) * 255.0 / 2.0).astype(int)
        imsave('output.png', image)

    sess.close()

















