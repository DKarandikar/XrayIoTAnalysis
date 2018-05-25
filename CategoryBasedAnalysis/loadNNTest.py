import tensorflow as tf
import random, os, pickle, sys
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

RANDOM_SEED = 83
MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_META_FILENAME = "model_normalized-1400.meta"
NUMBER_HIDDEN_NODES = 20

DATA_FILENAME = "normalized.csv"
NUMBER_COLUMNS = 56
NP_SAVE = True

try:
    if sys.argv[1] == "incOnly":
        NUMBER_COLUMNS = 20
        DATA_FILENAME = "onlyIncoming.csv"
        NP_SAVE = False
        MODEL_META_FILENAME = "model_onlyIncoming-400.meta"
        print("Incoming Packets Model")
except:
    print("Except")
    pass


FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  DATA_FILENAME)

PICKLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "liveCapFiles")

tf.set_random_seed(RANDOM_SEED)

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2, keep_prob):
    """
    Forward-propagation
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    relu = tf.nn.relu(h)
    dropout = tf.nn.dropout(relu, keep_prob)
    yhat = tf.matmul(dropout, w_2)  # The \varphi function
    return yhat

def confusionMatrix(real, pred):
    num_classes = np.max(real)
    matrix = np.zeros((num_classes, num_classes))
    #print(matrix.shape)
    for x in range(len(pred)):
        matrix[pred[x]-1, real[x]-1] += 1
    return matrix

def get_data():
    """ Read the csv data set and split them into training and test sets """
    
    df = pd.read_csv(FILE_PATH,
            usecols = [x for x in range(2,NUMBER_COLUMNS)],
            header=None)
    d = df.values

    l = pd.read_csv(FILE_PATH,
            usecols = [1], 
            header = None)
    labels = l.values

    data = np.float32(d)
    target = labels.flatten()

    #print(data.shape)

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
    #print(train_y)
    return train_X, test_X, train_y, test_y


def main(): 
    imported_meta = tf.train.import_meta_graph(os.path.join(MODELS_PATH, MODEL_META_FILENAME))  
    
    #print_tensors_in_checkpoint_file(file_name=os.path.join(MODELS_PATH, "iter_model-15500"), tensor_name = '', all_tensors=False)
    
    with tf.Session() as sess:
        imported_meta.restore(sess, os.path.join(MODELS_PATH, MODEL_META_FILENAME.split(".")[0]))
        graph = tf.get_default_graph()

        w1 = graph.get_tensor_by_name("weights1:0")
        w2 = graph.get_tensor_by_name("weights2:0")

        train_X, test_X, train_y, test_y = get_data()

        #np.set_printoptions(threshold=np.inf)
        #print(test_X)

        # Layer's sizes
        x_size = train_X.shape[1]   # Number of input nodes 
        h_size = NUMBER_HIDDEN_NODES                # Number of hidden nodes
        y_size = train_y.shape[1]   # Number of outcomes
         
        X = tf.placeholder("float", shape=[None, x_size])
        y = tf.placeholder("float", shape=[None, y_size])

        keep_prob = 1
        predict = tf.argmax(forwardprop(X, w1, w2, keep_prob), axis=1)

        final_predict = sess.run(predict, feed_dict={X: test_X})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: test_X}))

        print(" train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (100. * train_accuracy, 100. * test_accuracy))

        #print(np.argmax(test_y, axis=1))
        #print(final_predict)

        print(confusionMatrix(np.argmax(test_y, axis=1), final_predict))

        weights1 = np.array(sess.run(w1))
        weights2 = np.array(sess.run(w2))
        if NP_SAVE:
            np.save(os.path.join(PICKLE_PATH, "weights1"), weights1)
            np.save(os.path.join(PICKLE_PATH, "weights2"), weights2)


if __name__ == '__main__':
    main()