import tensorflow as tf

if tf.__version__.split('.')[0] == '2':
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import time
import os

from tensorflow.examples.tutorials.mnist import input_data


def RNN(x, weights, biases, type=1):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, nInput])
    x = tf.split(x, nSteps, 0)  # configuring so you can get it as needed for the 28 pixels

    # find which lstm to use in the documentation
    if type == 1:
        lstmCell = rnn_cell.BasicLSTMCell(nHidden, forget_bias=1.0) # Use LSTM
    elif type == 2:
        lstmCell = rnn_cell.GRUCell(nHidden)                        # Use GRU
    elif type == 3:
        lstmCell = rnn_cell.BasicRNNCell(nHidden)                   # Use RNN
    # elif type == 4:
    #     lstmCell = rnn.BasicRNNCell(nHidden)                        # Use LSTM and GRU

    # for the rnn where to get the output and hidden state#for the rnn where to get the output and hidden state
    outputs, states = rnn.static_rnn(lstmCell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


# optimization
# create the cost, optimization, evaluation, and accuracy
def optimization(label, logit):
    # for the cost softmax_cross_entropy_with_logits seems really good
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit))
    optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)
    correctPred = tf.equal(tf.argmax(logit, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32), name='accuracy')
    return cost, optimizer, accuracy


def training(dataset, x, y, cost, optimizer, accuracy, storeDir="./results/"):
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        summary_op = tf.summary.merge_all()  # Build the summary operation based on the TF collection of Summaries.
        summary_writer = tf.summary.FileWriter(storeDir, sess.graph)  # Instantiate a SummaryWriter to output summaries and the Graph.

        saver = tf.train.Saver()  # Create a saver for writing training checkpoints.
        step = 1

        start_time = time.time()  # start timing
        sess.run(init)
        while step * batchSize < trainingIters:
            batchX, batchY = dataset.train.next_batch(batchSize)  # mnist has a way to get the next batch
            batchX = batchX.reshape((batchSize, nSteps, nInput))

            feedDataTrain = {x: batchX, y: batchY}
            sess.run(optimizer, feed_dict=feedDataTrain)

            if step % displayStep == 0:
                checkpoint_file = os.path.join(storeDir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=step)

                acc = accuracy.eval(feed_dict=feedDataTrain)
                loss = cost.eval(feed_dict=feedDataTrain)
                print("Iter " + str(step * batchSize) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                      ", Training Accuracy= {:.5f}".format(acc))

                # Update the events file which is used to monitor the training (in this case,
                # only the training loss is monitored)
                summary_str = sess.run(summary_op, feed_dict=feedDataTrain)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            step += 1

        stop_time = time.time()
        print('Optimization finished')
        print('The training takes %g seconds to finish' % (stop_time - start_time))
        testData = dataset.test.images.reshape((-1, nSteps, nInput))
        testLabel = dataset.test.labels
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: testData, y: testLabel}))


learningRate = 1e-3
trainingIters = 100000
batchSize = 50
displayStep = 200

nInput = 28  # we want the input to take the 28 pixels
nSteps = 28  # every 28
nHidden = 32  # number of neurons for the RNN
nClasses = 10  # this is MNIST so you know


def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # call mnist function
    cell = ["LSTM", "GRU", "RNN"]

    i = 1

    x = tf.placeholder('float', [None, nSteps, nInput])
    y = tf.placeholder('float', [None, nClasses])
    weights = {'out': tf.Variable(tf.random_normal([nHidden, nClasses]))}
    biases = {'out': tf.Variable(tf.random_normal([nClasses]))}

    pred = RNN(x, weights, biases, i)
    cost, optimizer, accuracy = optimization(y, pred)
    tf.summary.scalar("Loss", cost)
    tf.summary.scalar("Accuracy", accuracy)

    training(dataset=mnist, x=x, y=y, cost=cost, optimizer=optimizer, accuracy=accuracy, storeDir="./results/" + str(i) + "_" + str(nHidden) + "/")


if __name__ == "__main__":
    main()
