import os, csv, _thread
import numpy as np
import pandas as pd
import pyshark
"""
Remember to add the column of 1s in front of it
"""

DEVICE_IP = "192.168.4.2"

BURST_PACKET_NO_CUTOFF = 60
BURST_TIME_INTERVAL = 1.0
FLOW_SIZE_CUTOFF = 10   # Minimum number of packets to be counted as a valid flow


# Setup the capture
capture = pyshark.LiveCapture(interface='wlan0', only_summaries=True, bpf_filter = "ip host " + DEVICE_IP)
print("Starting capture")


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

def addBiases(data):
    """ Adds a column of 1s to data"""
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data
    return all_X

def predictBurst(burst):
   
    flowStatistics = []

    # Get all IP sources and dests

    srcdest = getIps(burst)

    # Get lengths of flows

    flowLengths = getFlowDict(srcdest, burst)

    # Get statistics for each flow

    flowStatistics.extend(getStatisticsFromDict(flowLengths, srcdest, flowLengths))

    data = np.array(flowStatistics, dtype='float32')

    weights1 = np.load(os.path.join(FILE_PATH, "echoModel", "echoPCAweights1.npy"))
    weights2 = np.load(os.path.join(FILE_PATH, "echoModel", "echoPCAweights2.npy"))

    pca = pickle.load( open( os.path.join(FILE_PATH, "echoModel", "echoPCA.p"), "rb" ) )
    preScaler = pickle.load( open( os.path.join(FILE_PATH, "echoModel", "preScaler.p"), "rb" ) )
    postScaler = pickle.load( open( os.path.join(FILE_PATH, "echoModel", "postScaler.p"), "rb" ) )

    normalized = preScaler.transform(data)
    principal = pca.transform(normalized)
    principalNormalized = postScaler.transform(principal)

    all_X = addBiases(principalNormalized)

    hiddenPre = np.matmul(all_X, weights1)
    hiddenPost = np.maximum(hiddenPre, 0)   # pylint: disable=maybe-no-member
    output = np.matmul(hiddenPost, weights2)
    #print(output)

    category = np.argmax(output)

    categoryNames = {1: "Time", 2: "Weather", 3: "Joke", 4: "Song Author", 5: "Conversion", 6: "Day of week", 7: "Timer", 8: "Shopping", 9: "Lights", 10: "Alarms"}
    
    print(categoryNames[category])
    print(output)
    

nextBurst = []
first = True

for packet in capture.sniff_continuously():
    if first:
        currentTime = float(packet.time)
        first = False
    else:
        if (float(packet.time) - currentTime) < BURST_TIME_INTERVAL:
            nextBurst.append(packet)
            print("Appending")
            currentTime = float(packet.time)
        else:
            if len(nextBurst) > BURST_PACKET_NO_CUTOFF:
                _thread.start_new_thread( predictBurst, (nextBurst, ))
            else:
                print("Burst Too Short")
            currentTime = float(packet.time)
            nextBurst = [packet]