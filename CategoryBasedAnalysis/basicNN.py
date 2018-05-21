import tensorflow as tf
import random, os
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split

RANDOM_SEED = random.randint(1,100)
RANDOM_SEED = 83
#print(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    drop_out = tf.nn.dropout(h, 0.75)
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

def model(X, w):
    
    yhat = tf.matmul(X, w)
    return yhat

def confusionMatrix(real, pred):
    num_classes = np.max(real)
    matrix = np.zeros((num_classes, num_classes))
    print(matrix.shape)
    for x in range(len(pred)):
        matrix[pred[x]-1, real[x]-1] += 1
    return matrix
        
    

def get_data():
    """ Read the csv data set and split them into training and test sets """
    NUMBER_COLUMNS = 56
    
    df=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "noNonBiDirect.csv"),usecols = [x for x in range(2,NUMBER_COLUMNS-1)],skiprows = [0],header=0)
    #df=pd.read_csv(r'C:\Users\danie\OneDrive\MSc_CS\Project\TensorFlow\noNoiseFive.csv',usecols = [4, 5],skiprows = [0],header=0)
    d = df.values
    l = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "noNonBiDirect.csv"),usecols = [1] ,skiprows=[0], header=0)
    labels = l.values

    data = np.float32(d)
    target = labels.flatten()

    #print(data.shape)

    data /= np.max(np.abs(data)+0.0000001, axis=0)

    print(data)

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    # Convert into one-hot vectors
    all_Y = np.zeros((target.size, target.max()+1))
    all_Y[np.arange(target.size), target] = 1
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)
    #print(train_y)
    return train_X, test_X, train_y, test_y

def main():
    train_X, test_X, train_y, test_y = get_data()

    #print(train_y)

    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes 
    h_size = 10                 # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    #w_1 = init_weights((x_size, h_size))
    #w_2 = init_weights((h_size, y_size))

    w = init_weights((x_size, y_size))

    # Forward propagation
    #yhat    = forwardprop(X, w_1, w_2)
    yhat = model(X, w)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(500):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: test_X, y: test_y}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    final_predict = sess.run(predict, feed_dict={X: test_X, y: test_y})
    final_train_predict = sess.run(predict, feed_dict={X: train_X, y: train_y})

    sess.close()
    print(RANDOM_SEED)
    #print(np.argmax(train_y, axis=1))
    #print(final_train_predict)

    print(np.argmax(test_y, axis=1))
    print(final_predict)

    print(confusionMatrix(np.argmax(test_y, axis=1), final_predict))

if __name__ == '__main__':
    main()
    """
    real = [1, 2, 3, 2, 1]
    pred = [1, 2, 2, 3, 1]

    print(confusionMatrix(real, pred))
    """