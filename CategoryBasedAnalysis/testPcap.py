import os, statistics, csv, pyshark
from scapy.all import *
import pandas as pd
import tensorflow as tf
import numpy as np


BURST_PACKET_NO_CUTOFF = 60
BURST_TIME_INTERVAL = 1.0
FLOW_SIZE_CUTOFF = 10   # Minimum number of packets to be counted as a valid flow

MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
TRAINED_MODEL_META = os.path.join(MODELS_PATH, "iter_model-7800.meta")
NUMBER_HIDDEN_NODES_USED = 15
NUMBER_CLASSES_USED = 8

GETINPUT = False
TEST_FILENAME = "AlexaDayofWeek1"

DEVICE_IP = "192.168.4.2"

FEATURES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "FlowFeatures.csv")
DF = pd.read_csv(FEATURES_FILE, usecols = [x for x in range(2,NUMBER_COLUMNS)], header=None)
NUMBER_COLUMNS = 56

def normaliseColumn(array, colNo):
    """
    Min-max normalise data in array (a N * 54 shape) w.r.t. max/min in FEATURES_FILE
    """
    values = array[:, colNo]
    
    normalized = (values - DF.iloc[:,colNo].min()) / (DF.iloc[:,colNo].max() - DF.iloc[:,colNo].min() + 0.000000000000000001)
    
    # if colNo == 2:
    #     print(DF.iloc[:,colNo-2])
    #     print(values)
    #     print(DF.iloc[:,colNo-2].min())
    #     print(DF.iloc[:,colNo-2].max())

    array[:, colNo] = normalized
    return array

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    drop_out = tf.nn.dropout(h, 0.75)
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

def getStatistics(listInts):
    """
    Get 18 statistical features out of a list of integers
    """
    result = []
    df = pd.DataFrame()
    df['data'] = listInts

    result.append(df['data'].min())
    result.append(df['data'].max())
    result.append(df['data'].mean())
    result.append(df['data'].mad())
    result.append(df['data'].std())
    result.append(df['data'].var())
    result.append(df['data'].skew())
    result.append(df['data'].kurtosis())
    for value in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        result.append(df['data'].quantile(q=value))
    result.append(len(listInts))

    return result

def getBursts(packets):
    """
    Get all valid Bursts out of a list of packets
    """
    validBursts = []
    nextPcap = []
    currentTime = float(packets[0].sniff_timestamp)

    for p in packets:
        if (float(p.sniff_timestamp) - currentTime) < BURST_TIME_INTERVAL:
            nextPcap.append(p)
            currentTime = float(p.sniff_timestamp)
        else:
            if len(nextPcap) > BURST_PACKET_NO_CUTOFF:
                validBursts.append(nextPcap)
            currentTime = float(p.sniff_timestamp)
            nextPcap = [p]

    if len(nextPcap) > BURST_PACKET_NO_CUTOFF:
        validBursts.append(nextPcap)
    
    return validBursts

def getIps(burst):
    """ Get a list of IPs out of a burst """
    srcdest = set()

    for p in burst:
        if 'IP' in p:
            try:
                source = str(p['ip'].src)
                destination = str(p['ip'].dst)
                srcdest.add((source, destination))
            except AttributeError:
                print("Attribute error")
        
        
    srcdest = list(srcdest)
    return srcdest

def getFlowDict(sourcedest, burst):
    """
    Get a dictionary of lists of lengths of packets in the burst
    Keys are the souce-destination pairs of IP addresses
    """
    flowDict = {}

    for pair in sourcedest:
            flowLens = []
            source = pair[0]
            dest = pair[1]

            for p in burst:
                if 'IP' in p:
                    try:
                        if str(p['ip'].src) == source and str(p['ip'].dst) == dest:
                            flowLens.append(int(p.length))
                    except AttributeError:
                        print("Attribute error")
            

            flowDict[pair] = (flowLens)
    
    return flowDict

def getStatisticsFromDict(flowDict, sourceDest, lengthDict):
    """
    Get a list of 54 element lists
    Each sub-list is made up of three sets of 18 statistics
    These are generated from lengths of packets to, from, and both for each pair of IPs
    """
    result = []
    done = []
    for pair in sourceDest:
        if pair not in done and ((pair[1], pair[0])) in sourceDest:
            if len(lengthDict[pair])>2 and \
                len(lengthDict[(pair[1], pair[0])]) > 2 and \
                len(lengthDict[(pair[1], pair[0])]) + len(lengthDict[pair]) > FLOW_SIZE_CUTOFF:

                res = getStatistics(lengthDict[pair])
                res2 = getStatistics(lengthDict[(pair[1], pair[0])])
                res3 = getStatistics(lengthDict[pair] + lengthDict[(pair[1], pair[0])])

                done.append((pair[1], pair[0]))

                row = []

                # Ensure data is added in the following order: OUT / IN / BOTH
                if pair[0] == DEVICE_IP:
                    row.extend(res)
                    row.extend(res2)
                else:
                    row.extend(res2)
                    row.extend(res)
                row.extend(res3)

                result.append(row)

    return result

def checkNan(listListFloats):
    """
    Checks a list of list of floats for any nan
    """
    result = []

    for flowStat in listListFloats:
        if any(math.isnan(x) for x in flowStat):
            continue
        else:
            result.append(flowStat)

    return result

def addBiases(data):
    """ Adds a column of 1s to data"""
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data
    return all_X

def modelPrediction(data):
    """ 
    Runs the model on data and outputs predictions
    Uses lots of the constants at the top of the file for model data
    """
    imported_meta = tf.train.import_meta_graph(TRAINED_MODEL_META)  
    
    with tf.Session() as sess:

        imported_meta.restore(sess, tf.train.latest_checkpoint(MODELS_PATH))
        graph = tf.get_default_graph()

        w1 = graph.get_tensor_by_name("Variable:0")
        w2 = graph.get_tensor_by_name("Variable_1:0")

        # Layer's sizes
        x_size = data.shape[1]                            # Number of input nodes 
        h_size = NUMBER_HIDDEN_NODES_USED                 # Number of hidden nodes
        y_size = NUMBER_CLASSES_USED                      # Number of outcomes
         
        X = tf.placeholder("float", shape=[None, x_size])

        predict = tf.argmax(forwardprop(X, w1, w2), axis=1)

        final_predict = sess.run(predict, feed_dict={X: data})

        values = sess.run(forwardprop(X, w1, w2), feed_dict={X: data})
 
        return(final_predict, values)

def printPrediction(predictions, values):
    """ 
    Prints a list of int predictions using Alexa category names 
    Also prints other possible close categories
    """
    categoryNames = {1: "Time", 2: "Weather", 3: "Joke", 4: "Song Author", 5: "Conversion", 6: "Day of week", 7: "Timer", 8: "Shopping"}

    for counter, prediction in np.ndenumerate(predictions):
        
        theseVals = values[counter]
        #print(theseVals)

        average = np.mean(theseVals)
        std = np.std(theseVals)
        k = (theseVals[prediction]*1.0 - average)/std 

        print(categoryNames[prediction] + " at %f sigma" % k)

        for alternative in theseVals:
            if alternative > 0 and alternative != theseVals[prediction]:
                index = np.where(theseVals == alternative)[0][0]
                
                altK = (alternative*1.0 - average)/std
                print("Alternative: " + categoryNames[index] + " at %f sigma" % altK)

        print("")
        

def main(fileName):

    # Get all bursts of sufficient size

    pkts =  pyshark.FileCapture(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pcaps", fileName))

    bursts = getBursts(pkts)

    # Seprate out all the flows and get stats 

    flowStatistics = []

    for burst in bursts:

        # Get all IP sources and dests

        srcdest = getIps(burst)

        # Get lengths of flows

        flowLengths = getFlowDict(srcdest, burst)

        # Get statistics for each flow

        flowStatistics.extend(getStatisticsFromDict(flowLengths, srcdest, flowLengths))

    # Check for Not a Number issues (probably unnecessary now)

    validFlows = checkNan(flowStatistics)

    # Normalise the data by columns

    data = np.array(validFlows, dtype='float32')

    for x in range(NUMBER_COLUMNS-2):
        data = normaliseColumn(data, x)

    # Setup the model

    all_X = addBiases(data)

    # Categorise using the trained model

    prediction, percentages = modelPrediction(all_X)

    # Print in friendly format

    printPrediction(prediction, percentages)
    


if __name__ == '__main__':
    if GETINPUT:
        filename = input("Type pcap filename")
    else:
        filename = TEST_FILENAME

    main(filename)