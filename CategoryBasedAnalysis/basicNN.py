import tensorflow as tf
import random, os
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split

NUMBER_COLUMNS = 56
DATA_FILENAME = "normalizedClasses56.csv"
DATA_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", DATA_FILENAME )

MODEL_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models",  "model_" + DATA_FILENAME.split(".")[0])
HIDDEN_NODES = 15
SAVE = False
SAVE_INTERVAL = 100

RANDOM_SEED = 83

tf.set_random_seed(RANDOM_SEED)

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    drop_out = tf.nn.dropout(h, 0.75)
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

def model(X, w):
    """ A simpler model for use sometimes """
    yhat = tf.matmul(X, w)
    return yhat

def confusionMatrix(real, pred):
    """ Constructs a confusion matrix out of real and pred"""
    num_classes = np.max(real)
    matrix = np.zeros((num_classes, num_classes))
    print(matrix.shape)
    for x in range(len(pred)):
        matrix[pred[x]-1, real[x]-1] += 1
    return matrix

def get_data():
    """ Read the csv data set and split them into training and test sets """

    df = pd.read_csv(DATA_FILE_PATH, usecols = [x for x in range(2,NUMBER_COLUMNS)], header=None)
    d = df.values

    l = pd.read_csv(DATA_FILE_PATH, usecols = [1], header = None)
    labels = l.values

    data = np.float32(d)
    target = labels.flatten()

    #print(data.shape)
    
    # Data should be already min-max normalised
    #data /= np.max(np.abs(data)+0.0000001, axis=0)

    #print(data)

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    # Convert into one-hot vectors
    all_Y = np.zeros((target.size, target.max()+1))
    all_Y[np.arange(target.size), target] = 1
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)
    return train_X, test_X, train_y, test_y

def main():
    
    train_X, test_X, train_y, test_y = get_data()

    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes , which is NUMBER_COLUMNS - 2 + 1
    h_size = HIDDEN_NODES                 # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))
    #w = init_weights((x_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    #yhat = model(X, w)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD
    with tf.Session() as sess:
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(50000):
            # Train with each example
            for i in range(len(train_X)):
                sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

            train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                    sess.run(predict, feed_dict={X: train_X, y: train_y}))
            test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                    sess.run(predict, feed_dict={X: test_X, y: test_y}))

            print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

            if (epoch + 1)%SAVE_INTERVAL == 0 and SAVE:
                saver.save(sess, MODEL_FILENAME, global_step=epoch+1)
                #print(sess.run(w_1))
                #print(sess.run(w_2))

        final_predict = sess.run(predict, feed_dict={X: test_X, y: test_y})
        final_train_predict = sess.run(predict, feed_dict={X: train_X, y: train_y})

    print("Seed: " + str(RANDOM_SEED))

    #print(np.argmax(test_y, axis=1))
    #print(final_predict)

    print(confusionMatrix(np.argmax(test_y, axis=1), final_predict))

if __name__ == '__main__':
    main()
