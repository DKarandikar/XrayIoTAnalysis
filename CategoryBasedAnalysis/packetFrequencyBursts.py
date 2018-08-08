"""
Displays a bar chart with average time between packets in each burst, per category 
"""

import os, sys, random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scapy.all import rdpcap

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "AutomatedCapture"))
import statisticProcessing


def getFiles():
    """ Gets all audio files in a list"""
    mypath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bursts")
    files = []
    for (_, _, filenames) in os.walk(mypath):
        files.extend(filenames)
        break
    return files

files = getFiles()
dct = defaultdict(list)
success = 0

for file in files:
    
    if os.path.getsize(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bursts", file)) < 30000:
        continue

    packets = rdpcap(os.path.join(os.path.dirname(__file__), "bursts", file))

    deltas = []

    currentTime = packets[0].time

    for packet in packets[1:]:
        try:
            deltas.append(packet.time - currentTime)

            currentTime = packet.time
        except TypeError:
            pass
    

    dct[statisticProcessing.getFlowClass(file)].append(np.mean(deltas))

    success += 1

    #print("Processed " + file)

classes = [str(i) for i in range(1,12)]

classAverages = []
classStds = []

for key in classes:
    classAverages.append(np.mean(dct[key]))
    classStds.append(np.std(dct[key]))


plt.bar(range(len(classAverages)), classAverages, yerr=classStds, align='center')
plt.xticks(range(len(classAverages)), classes)

plt.xlabel("Alexa Command Category")
plt.ylabel("Average time between packets for all bursts (s)")

plt.show()