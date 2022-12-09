import time
from scipy import misc
from matplotlib.pyplot import imread
import numpy as np
import tensorflow as tf
import os
import random
import matplotlib.pyplot as plt
import matplotlib as mp
from skimage import transform

if tf.__version__.split('.')[0] == '2':
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    # print(tf.__version__)
import tensorflow as tf2
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --------------------------------------------------
# setup

def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    initial = tf.truncated_normal(shape, stddev=0.1)
    W = tf.Variable(initial)
    return W


def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    initial = tf.constant(0.1, shape=shape)
    b = tf.Variable(initial)
    return b


def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return h_conv


def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return h_max


## ==========================================================================================
## Load Data
def loadData():
    '''
    :param ntrain: training amount
    :param ntest: testing amount
    :param nclass: total label
    :param imsize: = width = length
    :param nchannels: input channel
    :return: Train: Inpout Data, Test: Input Data, LTrain: Learning Data, LTest: Learning Data
    '''
    Train = np.zeros((ntrain * nclass * rotation, imsize, imsize, nchannels))
    Test = np.zeros((ntest * nclass * rotation, imsize, imsize, nchannels))
    LTrain = np.zeros((ntrain * nclass * rotation, nclass))
    LTest = np.zeros((ntest * nclass * rotation, nclass))

    itrain = -1
    itest = -1
    for iclass in range(0, nclass):
        for isample in range(0, ntrain):
            path = './CIFAR10/Train/%d/Image%05d.png' % (iclass, isample)
            # for i in range(rotation):
            im = imread(path)  # 28 by 28
            # im = transform.rotate(im, angle=angle * i, cval=255)
            im = im.astype(float) / 255
            itrain += 1
            Train[itrain, :, :, 0] = im
            LTrain[itrain, iclass] = 1  # 1-hot lable
        for isample in range(0, ntest):
            path = './CIFAR10/Test/%d/Image%05d.png' % (iclass, isample)
            # for i in range(rotation):
            im = imread(path)  # 28 by 28
            # im = transform.rotate(im, angle=angle * i, cval=255)
            im = im.astype(float) / 255
            itest += 1
            Test[itest, :, :, 0] = im
            LTest[itest, iclass] = 1  # 1-hot lable
    return Train, Test, LTrain, LTest


## ==========================================================================================
# create your model
def model(tf_data, tf_output):
    # x_image = tf.reshape(data, [-1, imsize, imsize, nchannels])
    def convolutionLayer(data, width, height, inputChannel, outputChannel):
        W_conv = weight_variable([width, height, inputChannel, outputChannel])
        b_conv = bias_variable([outputChannel])
        z_conv = conv2d(data, W_conv) + b_conv
        h_conv = tf.nn.relu(z_conv)

        # Max pooling layer subsampling by 2
        h_pool = max_pool_2x2(h_conv)
        return W_conv, b_conv, z_conv, h_conv, h_pool

    def fullyConnectLayer(data, inputChannel, outputChannel):
        W_fc = weight_variable([inputChannel, outputChannel])
        b_fc = bias_variable([outputChannel])
        z_fc = tf.matmul(data, W_fc) + b_fc
        return W_fc, b_fc, z_fc

    ## First Layer
    # Convolutional layer with kernel 5x5 and 32 filter maps followed by ReLU
    W_conv1, b_conv1, z_conv1, h_conv1, h_pool1 = convolutionLayer(tf_data, 5, 5, nchannels, 32)

    ## Second Layer
    # Convolutional layer with kernel 5x5 and 64 filter maps followed by ReLU
    W_conv2, b_conv2, z_conv2, h_conv2, h_pool2 = convolutionLayer(h_pool1, 5, 5, 32, 64)

    ## Third Layer
    # Fully Connected layer that has input 7x7x64 and output 1024
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    W_fc1, b_fc1, z_fc1 = fullyConnectLayer(h_pool2_flat, 7 * 7 * 64, 1024)
    h_fc1 = tf.nn.relu(z_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## Forth Layer
    # Fully Connected layer that has input 1024 and output 10
    W_fc2, b_fc2, z_fc2 = fullyConnectLayer(h_fc1_drop, 1024, 10)
    # h_fc2 = tf.nn.relu(z_fc2)
    # Softmax Layer(Regression + Nonlinearity)
    y_output = tf.nn.softmax_cross_entropy_with_logits(labels=tf_output, logits=z_fc2)

    ## ==========================================================================================
    ## loss
    # set up the loss, evaluation, and accuracy
    cross_entropy = tf.reduce_mean(y_output)
    correct_prediction = tf.equal(tf.argmax(z_fc2, 1), tf.argmax(tf_output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    tf.summary.scalar("Loss", cross_entropy)
    tf.summary.scalar("Accuracy", accuracy)

    ## ==========================================================================================
    ## optimization
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    # optimizer = tf.train.MomentumOptimizer(learning_rate, .5).minimize(cross_entropy)
    # optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy)

    return cross_entropy, accuracy, optimizer, keep_prob, W_conv1, W_conv2, h_conv1, h_conv2


## ==========================================================================================
## Train
def training(tf_data, tf_labels, keep_prob, Train, LTrain, sess, cross_entropy, accuracy, optimizer,
             summary_op, summary_writer):
    # setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
    batch_xs = np.zeros([batchsize, imsize, imsize, nchannels])
    batch_ys = np.zeros([batchsize, nclass])  # setup as [batchsize, how many classes]

    saver = tf.train.Saver()  # Create a saver for writing training checkpoints.
    # the maximum iterations. After max_step iterations, the training will stop no matter what
    for i in range(max_step):  # try a small iteration size once it works then continue
        perm = np.arange(ntrain * nclass)
        np.random.shuffle(perm)
        for j in range(batchsize):
            batch_xs[j, :, :, :] = Train[perm[j], :, :, :]
            batch_ys[j, :] = LTrain[perm[j], :]

        feedData = {tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5}
        # calculate loss and train accuracy and print it
        loss = cross_entropy.eval(feed_dict=feedData)
        train_accuracy = accuracy.eval(feed_dict=feedData)
        if i % 100 == 0:
            checkpoint_file = os.path.join('./results/', 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=i)

            print("Step %d, Loss: %g, Training accuracy: %g" % (i, loss, train_accuracy))

            # Update the events file which is used to monitor the training (in this case,
            # only the training loss is monitored)
            summary_str = sess.run(summary_op, feed_dict=feedData)
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()

        optimizer.run(feed_dict=feedData)  # dropout only during training


## ==========================================================================================
## Test
def plotFigure(layer_weights):
    for i in range(32):
        plt.subplot(4, 8, i + 1)
        plt.imshow(layer_weights[:, :, 0, i], cmap="gray")
        plt.title("Filter" + str(i + 1))
        plt.axis("off")
    plt.show()


ntrain, ntest = 1000, 100  # per class
nclass = 10  # number of classes
imsize = 28
nchannels = 1
batchsize = 256
max_step = 3000
learning_rate = 1e-3
rotation = 1
angle = -0

def main():
    # Load Data
    Train, Test, LTrain, LTest = loadData()

    # Create Model
    sess = tf.InteractiveSession()
    # tf variable for the data, remember shape is [None, width, height, numberOfChannels]
    tf_data = tf.placeholder(tf.float32, shape=[None, imsize, imsize, nchannels])
    tf_labels = tf.placeholder(tf.float32, shape=[None, nclass])  # tf variable for labels
    cross_entropy, accuracy, optimizer, keep_prob, W_conv1, W_conv2, h_conv1, h_conv2 = model(tf_data, tf_labels)

    # tf.summary.scalar(cross_entropy.op.name, cross_entropy)  # Add a scalar summary for the snapshot loss.
    summary_op = tf.summary.merge_all()  # Build the summary operation based on the TF collection of Summaries.
    summary_writer = tf.summary.FileWriter('./results/',
                                           sess.graph)  # Instantiate a SummaryWriter to output summaries and the Graph.

    sess.run(tf.initialize_all_variables())

    start_time = time.time()  # start timing
    training(tf_data=tf_data, tf_labels=tf_labels, keep_prob=keep_prob,
             Train=Train, LTrain=LTrain,
             sess=sess,
             cross_entropy=cross_entropy, accuracy=accuracy, optimizer=optimizer,
             summary_op=summary_op, summary_writer=summary_writer)
    stop_time = time.time()
    print('The training takes %g seconds to finish' % (stop_time - start_time))

    feedData = {tf_data: Test, tf_labels: LTest, keep_prob: 1.0}
    print("test accuracy %g" % accuracy.eval(feed_dict=feedData))

    # Calculate the statistics of the activations in the convolutional layers on test images.
    activation1 = h_conv1.eval(feed_dict=feedData)
    activation2 = h_conv2.eval(feed_dict=feedData)
    print("Activation1: Mean %g, Variance %g" % (np.mean(np.array(activation1)), np.var(np.array(activation1))))
    print("Activation2: Mean %g, Variance %g" % (np.mean(np.array(activation2)), np.var(np.array(activation2))))

    plotFigure(W_conv1.eval())
    plotFigure(W_conv2.eval())

    sess.close()


if __name__ == "__main__":
    main()
