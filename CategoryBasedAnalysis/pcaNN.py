import tensorflow as tf
import random, os, pickle, sys, time
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split

## Constants to control the whole process  ==========================

HYPERSEARCH = True # Whether or not to do hyperparameter search 
REGULARIZE = True # Apply regularization
SAVE = True # Wether to save the model
PICKLE_ACCURACIES = True # Whether to pickle accuracies

# Whether or not to continue training from a previous model 
CONTINUE = False
META_TO_CONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models",  "model_normalizedPCAGoogleWeather-1500.meta")
EPOCH_SO_FAR = 1500
if not CONTINUE:
    EPOCH_SO_FAR = 0

## Constants about the data ===========================

NUMBER_COLUMNS = 26 # Number of features plus 2 
DATA_FILENAME = "normalizedPCAWeatherDeltas.csv"

COMBINE_LIGHTS = False  # Combine categories 9 and 10 and renumber any after it 
ONLY_KEY_CATEGORIES = False # Only Time, Shopping, Joke, LightsCombined and Alarms, only use with the above
WEATHER_CATEGORIES = True # Use if categories are 21, 22, 23, 24 ; should only be used with the above two false 

RANDOMISE_DATA = False # Randomise all classes to see if too much structure, only do with only_key and combine on or with Weather, and those off 

## Constants about the model =========================

HIDDEN_NODES = 20 
SAVE_INTERVAL = 500
TOTAL_EPOCHS = 1500

REGULARIZER = "L1" # Either "L1" or "L2" is the default
SCALE = 0.005 # Scale for regularizer 
REGULARIZE_WEIGHTS = "" # Either "Both" or default is only w_2 

## Other ===========================

RANDOM_SEED = 83
tf.set_random_seed(RANDOM_SEED)

DATA_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataProc", "data", DATA_FILENAME )
MODEL_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models",  "model_" + DATA_FILENAME.split(".")[0])


# This isn't used anymore 
onlyIncoming = False
try:
    if sys.argv[1] == "incOnly":
        onlyIncoming = True
        SAVE=True
        MODEL_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models",  "model_" + "incOnly")
        PICKLE_ACCURACIES = False
        HIDDEN_NODES = 10
        print("Incoming Packets Model")
except:
    pass


def init_weights(shape, varName):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights, name = varName)

def forwardprop(X, w_1, w_2, keep_prob):
    """
    Forward-propagation
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  
    relu = tf.nn.relu(h)
    dropout = tf.nn.dropout(relu, keep_prob)
    yhat = tf.matmul(dropout, w_2)  
    return yhat

def basicModel(X, w):
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
    if not onlyIncoming:
        df = pd.read_csv(DATA_FILE_PATH, usecols = [x for x in range(2,NUMBER_COLUMNS)], header=None)
    else:
        # Data is added in the following order: OUT / IN / BOTH
        df = pd.read_csv(DATA_FILE_PATH, usecols = [x for x in range(20,NUMBER_COLUMNS-18)], header=None)
    d = df.values

    l = pd.read_csv(DATA_FILE_PATH, usecols = [1], header = None)
    labels = l.values

    data = np.float32(d)
    target = labels.flatten()
    
    # Combine 9 and 10 and renumber all after
    if COMBINE_LIGHTS:
        for index, value in enumerate(target):
            if value == 10:
                target[index] = 9    
            if value > 10:
                target[index] = value - 1

    if ONLY_KEY_CATEGORIES:

        # Add up how many data points are in these categories so we can shape target
        totalKeyCategories = 0
        for x in target:
            if x in [9, 1, 10, 8, 3]:
                totalKeyCategories += 1

        newData = np.zeros(shape=(totalKeyCategories, NUMBER_COLUMNS-2), dtype=float)
        newTarget = np.zeros(shape = (totalKeyCategories), dtype=int)

        tick = 0 # Counts along newData, can't use index cause points are skipped 

        for index, value in enumerate(target):
            if value in [9, 1, 10, 8, 3]:
                
                if RANDOMISE_DATA:
                    newTarget[tick] = random.choice([9, 1, 10, 8, 3])
                else:
                    newTarget[tick] = value

                newData[tick, :] = data[index, :]
                tick += 1

        data = newData
        target = newTarget

    # If categories are 21, 22, 23, 24, re-label to 1,2,3,4 so that tensorflow is happy when converting to one-hot
    if WEATHER_CATEGORIES:
        newTarget = np.zeros(shape = target.shape, dtype=int)
        for index, value in enumerate(target):
            if RANDOMISE_DATA:
                newTarget[index] = random.choice([1, 2, 3, 4])
            else:
                newTarget[index] = value - 20

        target = newTarget

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
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_Y, test_size=0.2, random_state=RANDOM_SEED)

    return train_X, test_X, train_y, test_y

def saveModelAndAccuracies(epoch, saver, sess, testAccuracy, trainAccuracy, saveNumber):
    """
    Save the model using saver in the sess
    Also if PICKLE_ACCURACIES, then pickle the test and train accuracies so far
    If saveNumber isn't false then number the models and accuracies 
    """
    if (epoch + 1)%SAVE_INTERVAL == 0 and SAVE:
        if saveNumber != False:
            saver.save(sess, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models",  "model_" + DATA_FILENAME.split(".")[0] + str(saveNumber)), global_step=epoch+1)
            if PICKLE_ACCURACIES:
                pickle.dump(testAccuracy, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'testAccuracies'+str(saveNumber)+ '.p'),'wb'))
                pickle.dump(trainAccuracy, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'trainAccuracies'+str(saveNumber)+ '.p'),'wb'))
        else:
            saver.save(sess, MODEL_FILENAME, global_step=epoch+1)
            if PICKLE_ACCURACIES:
                pickle.dump(testAccuracy, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'testAccuracies.p'),'wb'))
                pickle.dump(trainAccuracy, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'trainAccuracies.p'),'wb'))

def regularise(cost, updates, w_1, w_2):
    """
    Adds regularisation as determined by global constants to w_1 and w_2
    Returns the adjusted cost and updates 
    """
    if REGULARIZE:
        if REGULARIZER == "L1":
            regularizer = tf.contrib.layers.l1_regularizer(
                scale=SCALE, scope=None
            )
        else:
            regularizer = tf.contrib.layers.l2_regularizer(
                scale=SCALE, scope=None
            )
        

        if REGULARIZE_WEIGHTS == "Both":
            regularization_penalty = tf.contrib.layers.apply_regularization(regularizer, [w_1, w_2])
        else:
            regularization_penalty = tf.contrib.layers.apply_regularization(regularizer, [w_2])

        cost = cost + regularization_penalty
        updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
        return cost, updates
    else:
        return cost, updates 

def runNN(printing=True, saveNumber = False):
    """
    Runs the whole neural network 
    If printing is False then it doesn't print test/train as going along or the final confusion matrix
    If saveNumber is an integer then models are numbered when saved, as are test train accuracies
        This is useful for hypersearch 
    """
    if CONTINUE:
        imported_meta = tf.train.import_meta_graph(META_TO_CONT)  

    # Run SGD
    with tf.Session() as sess:

        #### Setting up the model ================================

        train_X, test_X, train_y, test_y = get_data()

        # Layer's sizes
        x_size = train_X.shape[1]   # Number of input nodes , which is NUMBER_COLUMNS - 2 + 1
        h_size = HIDDEN_NODES                 # Number of hidden nodes
        y_size = train_y.shape[1]   # Number of outcomes

        # Weight initializations
        if CONTINUE:
            imported_meta.restore(sess, META_TO_CONT.split(".")[0])
            graph = tf.get_default_graph()

            w_1 = graph.get_tensor_by_name("weights1:0")
            w_2 = graph.get_tensor_by_name("weights2:0")
        else:
            w_1 = init_weights((x_size, h_size), "weights1")
            w_2 = init_weights((h_size, y_size), "weights2")
    
        # Symbols
        X = tf.placeholder("float", shape=[None, x_size])
        y = tf.placeholder("float", shape=[None, y_size])

        # Forward propagation
        keep_prob = tf.placeholder_with_default(1.0, shape=())
        yhat    = forwardprop(X, w_1, w_2, keep_prob)
        predict = tf.argmax(yhat, axis=1)

        # Backward propagation
        cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
        updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

        cost, updates = regularise(cost, updates, w_1, w_2)

        # Setup accuracies if some are already stored 

        if CONTINUE and PICKLE_ACCURACIES:
            trainAccuracy = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'trainAccuracies.p'),'rb'))
            testAccuracy = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'testAccuracies.p'),'rb'))
        else:
            trainAccuracy = []
            testAccuracy = []


        #### Now running the model ================================
    

        saver = tf.train.Saver()
        if not CONTINUE:
            init = tf.global_variables_initializer()
            sess.run(init)

        startTime = time.time()

        

        for epoch in range(EPOCH_SO_FAR, EPOCH_SO_FAR + TOTAL_EPOCHS):
            
            # Train with each example
            for i in range(len(train_X)):
                sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1], keep_prob: 0.9})

            # Get accuracies without dropout
            train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                    sess.run(predict, feed_dict={X: train_X, y: train_y, keep_prob: 1}))
            test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                    sess.run(predict, feed_dict={X: test_X, y: test_y, keep_prob: 1}))

            timeLeft = ((time.time()-startTime) * (TOTAL_EPOCHS -epoch) * 1.0/(epoch - EPOCH_SO_FAR+1) *1.0)

            if printing:
                print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                    % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
                
                print("Estimated time left is %.2f seconds" % timeLeft )
            else: 
                print ("                                                                   ", end="\r"),
                print ("At %.5f with %.2f seconds left" % (epoch*1.0/TOTAL_EPOCHS, timeLeft), end="\r"),

            testAccuracy.append(100. * train_accuracy)
            trainAccuracy.append(100. * test_accuracy)

            saveModelAndAccuracies(epoch, saver, sess, testAccuracy, trainAccuracy, saveNumber)
            

        final_predict = sess.run(predict, feed_dict={X: test_X, y: test_y})
        final_train_predict = sess.run(predict, feed_dict={X: train_X, y: train_y})

    if printing:
        print("Seed: " + str(RANDOM_SEED))
        #print(np.argmax(test_y, axis=1))
        #print(final_predict)
        print(confusionMatrix(np.argmax(test_y, axis=1), final_predict))

    return (trainAccuracy[-1], testAccuracy[-1])

def main():
    if HYPERSEARCH:
        #SAVE = False
        #PICKLE_ACCURACIES = False

        results = []

        hiddenLayers = [10, 15, 20]
        regular = ["L1", "L2"]
        scales = [0.01, 0.005, 0.001, 0.0005, 0.0001]
        weights = ["Both", "Only W2"]

        s = [hiddenLayers, regular, scales, weights]

        allOptions = list(itertools.product(*s))

        allOptions = [(10, "L1", 0.0001, "Only W2"),
                        (15, "L1", 0.01, "Only W2"),
                        (15, "L1", 0.0001, "Both"),
                        (20, "L2", 0.0001, "Only W2"),
                        (10, "L1", 0.0005, "Both")]

        for index, option in enumerate(allOptions):

            HIDDEN_NODES = option[0]
            REGULARIZER = option[1]
            SCALE = option[2]
            REGULARIZE_WEIGHTS = option[3] # Either "Both" or default is only w_2 

            X = runNN(printing = False, saveNumber = index+1)
            train = X[0]
            test = X[1]

            results.append((train, test, option[0], option[1], option[2], option[3]))
            print( "%d of %d: Train accuracy of %.2f%% and test of %.2f%% with %d hidden nodes and a %s regularizer with scale of %f applied to %s"  % (index, len(allOptions), train, test, option[0], option[1], option[2], option[3]))
        
        print("Full Results:")
        for resultTuple in results:
            print( "Train accuracy of %.2f%% and test of %.2f%% with %d hidden nodes and a %s regularizer with scale of %f applied to %s"  % resultTuple)
    else:
        runNN()

if __name__ == '__main__':
    main()
