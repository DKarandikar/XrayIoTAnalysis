import os, datetime, csv
import pandas as pd
from scapy.all import IP

DEVICE_IP = "192.168.4.2"
BURST_PACKET_NO_CUTOFF = 60
BURST_TIME_INTERVAL = 0.5
FLOW_SIZE_CUTOFF = 20   # Minimum number of packets to be counted as a valid flow
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))

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
    currentTime = float(packets[0].time)

    for p in packets:
        if (float(p.time) - currentTime) < BURST_TIME_INTERVAL:
            nextPcap.append(p)
            currentTime = float(p.time)
        else:
            if len(nextPcap) > BURST_PACKET_NO_CUTOFF:
                validBursts.append(nextPcap)
            currentTime = float(p.time)
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
                source = str(p[IP].src)
                destination = str(p[IP].dst)
                srcdest.add((source, destination))
            except IndexError:
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
                        if str(p[IP].src) == source and str(p[IP].dst) == dest:
                            flowLens.append(int(p.len) + 14)    # The +14 is to deal with a scapy/pyshark mismatch
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

def getFlowClass(filename):
    if "Timers" in filename:
        flowClass = "7"
    elif "Weather" in filename:
        flowClass = "2"
    elif "Joke" in filename:
        flowClass = "3"
    elif "Sings" in filename:
        flowClass = "4"
    elif "Conversion" in filename:
        flowClass = "5"
    elif "Time" in filename:
        flowClass = "1"
    elif "DayofWeek" in filename:
        flowClass = "6"
    elif "Shopping" in filename:
        flowClass = "8"
    elif "LightsOnOff" in filename:
        flowClass = "9"
    elif "LightsBrightDim" in filename:
        flowClass = "10"
    elif "Alarms" in filename:
        flowClass = "11"
    else:
        flowClass = "0"

    return flowClass

def getCSVWriter(timeData=False):
    ### Setup csv file
    if timeData:
        csvPath = os.path.join(FILE_PATH, "timeData")
    else:
        csvPath = os.path.join(FILE_PATH, "data")
    if not os.path.exists(csvPath):
        os.makedirs(csvPath)

    dataFile = "Flowfeatures.csv"

    if timeData:
        dataFile = "FlowFeaturesTime.csv"

    newFile = not os.path.isfile(os.path.join(csvPath, dataFile))
    files =[]

    if newFile:
        output = open(os.path.join(csvPath,  dataFile),'a', newline='')
        writer = csv.writer(output)

    else:
        with open(os.path.join(csvPath,  dataFile), 'r') as csvFile:
            mycsv = csv.reader(csvFile)
                

        output = open(os.path.join(csvPath,  dataFile),'a', newline='')
        writer = csv.writer(output)

    return writer

def saveStatistics(listListFloats, filename, bNo):
    
    now = datetime.datetime.now()
    date = "%d-%d-%d" % (now.day, now.month, now.year)
    packetsPath = os.path.join(FILE_PATH, "savedPackets" + date)
    counter = 0
    while True:
        if os.path.isfile(os.path.join(packetsPath, filename + str(counter) + ".pcap")):
            counter += 1
        else:
            break

    counter -=1

    fCounter = 0
    for listFloats in listListFloats:
        row = []
        row.append(filename + str(counter) + "On" + date + "Burst" + str(bNo) + "Flow" + str(fCounter))

        classNumber = getFlowClass(filename)

        row.append(classNumber)

        row.extend(listFloats)

        writer = getCSVWriter()

        writer.writerow(row)

        fCounter += 1

def processPackets(packets, filename, ret=False):
    bursts = getBursts(packets)

    # Seprate out all the flows and get stats 

    flowStatistics = []

    burstNo = 0

    allBursts = []

    for burst in bursts:

        # Get all IP sources and dests

        srcdest = getIps(burst)

        # Get lengths of flows

        flowLengths = getFlowDict(srcdest, burst)

        # Get statistics for each flow

        flowStatistics = getStatisticsFromDict(flowLengths, srcdest, flowLengths)

        if not ret:
            saveStatistics(flowStatistics, filename, burstNo)
        else:
            allBursts.append(flowStatistics)

        burstNo += 1

    if ret:
        return allBursts
