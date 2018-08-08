"""
Important Note: This is specialised to Amazon Echo (Alexa) right now
Extracts statistical features from a folder full of pcaps each of which is a burst
"""

import os, sys
from scapy.all import *
import statistics, csv, pyshark
import pandas as pd

DEVICE_IP = "192.168.4.2"

FLOW_SIZE_CUTOFF = 10   # Minimum number of packets to be counted as a valid flow

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

### Get names of all burst files

f = []
for (d, dn, filenames) in os.walk(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bursts") ):
    f.extend(sorted(filenames))
    break

#print(f)


### Setup csv file

newFile = not os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Flowfeatures.csv"))
files =[]

if newFile:
    output = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "Flowfeatures.csv"),'a', newline='')
    writer = csv.writer(output)

else:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "Flowfeatures.csv"), 'r') as csvFile:
        mycsv = csv.reader(csvFile)
        for row in mycsv:
            if row:
                # Get the filename before Flow is appended
                files.append(row[0].split("Flow")[0])
                

    output = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "Flowfeatures.csv"),'a', newline='')
    writer = csv.writer(output)


### Extract features

for file in f:
    print("Extracting features: " + file)

    if not newFile:
        # Check files against the name before .pcap
        if file.split(".")[0] in files:
            print("Already Done")
            continue

    
    # Class
    ## Class labels are specialised to Alexa currently

    if "Timers" in file and os.path.getsize(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bursts", file)) > 30000:
        flowClass = "7"
    elif "Weather" in file and os.path.getsize(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bursts", file)) > 30000:
        flowClass = "2"
    elif "Joke" in file and os.path.getsize(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bursts", file)) > 30000:
        flowClass = "3"
    elif "Sings" in file and os.path.getsize(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bursts", file)) > 30000:
        flowClass = "4"
    elif "Conversion" in file and os.path.getsize(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bursts", file)) > 30000:
        flowClass = "5"
    elif "Time" in file and os.path.getsize(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bursts", file)) > 30000:
        flowClass = "1"
    elif "DayofWeek" in file and os.path.getsize(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bursts", file)) > 30000:
        flowClass = "6"
    elif "Shopping" in file and os.path.getsize(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bursts", file)) > 30000:
        flowClass = "8"
    elif "LightsOnOff" in file and os.path.getsize(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bursts", file)) > 30000:
        flowClass = "9"
    elif "LightsBrightDim" in file and os.path.getsize(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bursts", file)) > 30000:
        flowClass = "10"
    elif "Alarms" in file and os.path.getsize(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bursts", file)) > 30000:
        flowClass = "11"
    else:
        print("Noise")
        continue

    # Flow Statistics

    pkts = pyshark.FileCapture(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bursts", file))

    # Get all IP sources and dests

    srcdest = set()
    

    for p in pkts:
        if 'IP' in p:
            try:
                source = str(p['ip'].src)
                destination = str(p['ip'].dst)
                srcdest.add((source, destination))
            except AttributeError:
                print("Attribute error")
    
    
    srcdest = list(srcdest)
    #print(srcdest)
    #print(len(srcdest))

    # Get lengths of flows
    # Lengths of packets for each direction and bi-directional

    flowLengths = {}

    for pair in srcdest:
        flowLens = []
        source = pair[0]
        dest = pair[1]

        for p in pkts:
            if 'IP' in p:
                try:
                    if str(p['ip'].src) == source and str(p['ip'].dst) == dest:
                        flowLens.append(int(p.length))
                except AttributeError:
                    print("Attribute error")

        flowLengths[pair] = flowLens

    # Now get statistics and print to a line for each flow
    # Each flow has a class and then all statistics for traffic in both directions sepearately and together
    # This is a total of 54 features per flow, plus name and class
    
    done = []
    counter = 1
    for pair in srcdest:
        if pair not in done and ((pair[1], pair[0])) in srcdest:
            if len(flowLengths[pair])>2 and \
                len(flowLengths[(pair[1], pair[0])]) > 2 and \
                len(flowLengths[(pair[1], pair[0])]) + len(flowLengths[pair]) > FLOW_SIZE_CUTOFF:

                res = getStatistics(flowLengths[pair])
                res2 = getStatistics(flowLengths[(pair[1], pair[0])])
                res3 = getStatistics(flowLengths[pair] + flowLengths[(pair[1], pair[0])])

                done.append((pair[1], pair[0]))

                row = []
                row.append(file.split(".")[0] + "Flow" + str(counter))
                row.append(flowClass)

                # Ensure data is added in the following order: OUT / IN / BOTH
                if pair[0] == DEVICE_IP:
                    row.extend(res)
                    row.extend(res2)
                else:
                    row.extend(res2)
                    row.extend(res)
                row.extend(res3)

                counter += 1

                writer.writerow(row)

        
    
output.close()
    



