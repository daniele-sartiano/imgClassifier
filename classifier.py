'''
AlexNet implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
AlexNet Paper (http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from Dataset import Dataset
import utils

import os
import tensorflow as tf
import numpy as np


IMG_SIZE = 30

LABELS_DICT = {
    'Cani': 0,
    'Cavalli': 1,
    'Alberi': 2,
    'Gatti': 3
}

BATCH_SIZE = 32

# Parameters
learning_rate = 0.001
training_iters = 200000
display_step = 20

# Network Parameters
n_input = IMG_SIZE**2
n_classes = 4 # MNIST total classes (0-9 digits)
n_channel = 3
dropout = 0.8 # Dropout, probability to keep units



# tf Graph input
#x = tf.placeholder(tf.float32, [None, n_input, n_channel])
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


def convLabels(imageDir):
    index_offset = np.arange(1) * len(LABELS_DICT)
    labels_one_hot = np.zeros((1, len(LABELS_DICT)))
    labels_one_hot.flat[index_offset + np.array([LABELS_DICT[imageDir]])] = 1
    return labels_one_hot[0]


#np.set_printoptions(threshold=np.nan)

def getDataset():
    with tf.Session() as session:
        tf.initialize_all_variables().run()

        for dirName in os.listdir(utils.image_dir):
            label = convLabels(dirName)
            path = os.path.join(utils.image_dir, dirName)
            for img in os.listdir(path):
                img_path = os.path.join(path, img)
                if os.path.isfile(img_path) and img.endswith('jpeg'):
                    img_bytes = tf.read_file(img_path)
                    #img_u8 = tf.image.decode_jpeg(img_bytes)
                    img_u8 = tf.image.decode_jpeg(img_bytes, channels=1)
                    img_u8_eval = session.run(img_u8)

                    image = tf.image.convert_image_dtype(img_u8_eval, dtype=tf.float32)
                    img_padded_or_cropped = tf.image.resize_image_with_crop_or_pad(image, IMG_SIZE, IMG_SIZE)
                    
                    #img_padded_or_cropped = tf.reshape(img_padded_or_cropped, shape=[IMG_SIZE*IMG_SIZE, 3])
                    img_padded_or_cropped = tf.reshape(img_padded_or_cropped, shape=[IMG_SIZE*IMG_SIZE])

                    yield img_padded_or_cropped.eval(), np.array(label)
    

def nextBatch(imgs, labels, step, batch_size):
    s = step*batch_size
    return imgs[s:s+batch_size], labels[s:s+batch_size]
    
# Create AlexNet model
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)

def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def alex_net(_X, _weights, _biases, _dropout):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, IMG_SIZE, IMG_SIZE, 1])

    # Convolution Layer
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
    # Max Pooling (down-sampling)
    pool1 = max_pool('pool1', conv1, k=2)
    # Apply Normalization
    norm1 = norm('norm1', pool1, lsize=4)
    # Apply Dropout
    norm1 = tf.nn.dropout(norm1, _dropout)

    # Convolution Layer
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    # Max Pooling (down-sampling)
    pool2 = max_pool('pool2', conv2, k=2)
    # Apply Normalization
    norm2 = norm('norm2', pool2, lsize=4)
    # Apply Dropout
    norm2 = tf.nn.dropout(norm2, _dropout)

    # Convolution Layer
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    # Max Pooling (down-sampling)
    pool3 = max_pool('pool3', conv3, k=2)
    # Apply Normalization
    norm3 = norm('norm3', pool3, lsize=4)
    # Apply Dropout
    norm3 = tf.nn.dropout(norm3, _dropout)

    # Fully connected layer
    dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') # Relu activation

    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation

    # Output, class prediction
    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, BATCH_SIZE])),
    'wc2': tf.Variable(tf.random_normal([3, 3, BATCH_SIZE, BATCH_SIZE*2])),
    'wc3': tf.Variable(tf.random_normal([3, 3, BATCH_SIZE*2, BATCH_SIZE*4])),
    'wd1': tf.Variable(tf.random_normal([4*4*BATCH_SIZE*4, BATCH_SIZE*16])),
    'wd2': tf.Variable(tf.random_normal([BATCH_SIZE*16, BATCH_SIZE*16])),
    'out': tf.Variable(tf.random_normal([BATCH_SIZE*16, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([BATCH_SIZE])),
    'bc2': tf.Variable(tf.random_normal([BATCH_SIZE*2])),
    'bc3': tf.Variable(tf.random_normal([BATCH_SIZE*4])),
    'bd1': tf.Variable(tf.random_normal([BATCH_SIZE*16])),
    'bd2': tf.Variable(tf.random_normal([BATCH_SIZE*16])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = alex_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 0

    imgs = []
    labels = []
    for img, label in getDataset():
        imgs.append(img)
        labels.append(label)


    print 'Dataset created'


    for epoch in xrange(training_iters):
        for step in xrange((len(imgs)/BATCH_SIZE) +1):

            batch_xs, batch_ys = nextBatch(imgs, labels, step, BATCH_SIZE)
            
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        
            if step % display_step == 0:
                print 'calculating accuracy'
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                print "Iter " + str(step*BATCH_SIZE) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)

    print "Optimization Finished!"
    # Calculate accuracy for 256 mnist test images

    #TODO: add test

    #print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.})
