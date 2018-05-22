import os, statistics, csv, pyshark
from scapy.all import *
import pandas as pd
import tensorflow as tf
import numpy as np

"""
Will need to get all bursts, then flows
then test on the NN
"""

BURST_PACKET_NO_CUTOFF = 60
BURST_TIME_INTERVAL = 1.0

MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
TRAINED_MODEL_META = os.path.join(MODELS_PATH, "iter_model-15500.meta")

NUMBER_COLUMNS = 56

DF = pd.read_csv( os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "noNonBiDirect.csv"), usecols = [x for x in range(4,NUMBER_COLUMNS)], header=None)

def normaliseColumn(array, colNo):
    values = array[:, colNo]
    
    normalized = (values - DF.iloc[:,colNo-2].min()) / (DF.iloc[:,colNo-2].max() - DF.iloc[:,colNo-2].min() + 0.000000000000000001)
    if colNo == 2:
        print(DF.iloc[:,colNo-2])
        print(values)
        print(DF.iloc[:,colNo-2].min())
        print(DF.iloc[:,colNo-2].max())

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

# Get a variety of statistics out of a list of Ints
def getStatistics(listInts):
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


def main(fileName):
    
    ### Get all bursts of sufficient size

    pkts = pyshark.FileCapture(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pcaps", fileName))
    validBursts = []
    nextPcap = []

    currentTime = float(pkts[0].sniff_timestamp)

    for p in pkts:
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

    ### Seprate out all the flows

    #print("Length is: " + str(len(validBursts)))

    flowStatistics = []

    # Get all IP sources and dests

    for burst in validBursts:

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

        print(srcdest)

        ### Get lengths of flows
        ### Lengths of packets for each direction and bi-directional

        flowLengths = {}
        counter = 0

        for pair in srcdest:
            flowLens = []
            source = pair[0]
            dest = pair[1]

            for p in burst:
                if 'IP' in p:
                    try:
                        if str(p['ip'].src == source and p['ip'].dst == dest):
                            flowLens.append(int(p.length))
                    except AttributeError:
                        print("Attribute error")

            # Here we look for the flipped source, dest to get bi-directional traffic
            testCount = 0
            for testPair in srcdest[:counter]:
                if testPair == (dest, source):
                    flowLengths[(source, dest, "both")] = flowLens + flowLengths[(dest, source)]
                testCount += 1
            
            counter += 1
            flowLengths[pair] = (flowLens)


        done = []
        counter = 0
        for pair in srcdest:
            if pair not in done:
                res = getStatistics(flowLengths[pair])
                res2 = getStatistics(flowLengths.get((pair[1], pair[0]), []))
                done.append((pair[1], pair[0]))
                res3 = getStatistics(flowLengths.get((pair[1], pair[0], "both"), []))

            row = []
            row.append(0)
            row.append(0)
            row.extend(res)
            row.extend(res2)
            row.extend(res3)

            flowStatistics.append(row)
                

    ### Remove any non bi-directional flows

    #print(flowStatistics)
    #print(len(flowStatistics))
    #print(len(validBursts))

    validFlows = []

    for flowStat in flowStatistics:
        if any(math.isnan(x) for x in flowStat):
            continue
        else:
            validFlows.append(flowStat)

    #pkts.close() 

    ### Normalise

    print(validFlows)

    data = np.array(validFlows, dtype='float32')[:, 2:55]

    for x in range(2,53):
        data = normaliseColumn(data, x)

    #print(data)

    #data /= np.max(np.abs(data)+0.0000001, axis=0)

    #np.set_printoptions(threshold=np.inf)
    #print(data)

    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    all_X[:, 1] = np.ones(N)
    all_X[:, 2] = np.ones(N)

    print(all_X)
    np.savetxt("test.txt", all_X)

    ### Categorise using the trained model

    imported_meta = tf.train.import_meta_graph(TRAINED_MODEL_META)  
    
    with tf.Session() as sess:
        imported_meta.restore(sess, tf.train.latest_checkpoint(MODELS_PATH))
        graph = tf.get_default_graph()

        w1 = graph.get_tensor_by_name("Variable:0")
        w2 = graph.get_tensor_by_name("Variable_1:0")

        # Layer's sizes
        x_size = 54                 # Number of input nodes 
        h_size = 15                 # Number of hidden nodes
        y_size = 8                  # Number of outcomes
         
        X = tf.placeholder("float", shape=[None, x_size])

        predict = tf.argmax(forwardprop(X, w1, w2), axis=1)

        final_predict = sess.run(predict, feed_dict={X: all_X})

        print(final_predict)


if __name__ == '__main__':
    #filename = input("Type pcap filename")
    filename = "AlexaTest2"
    main(filename)