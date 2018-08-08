"""
Splits pickled packets into bursts, but again this not used anymore 
"""
import os,  inspect, json
from scapy.all import rdpcap, wrpcap, IP

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

BURSTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bursts")
TIME_INTERVAL = 1.0

if not os.path.exists(BURSTS_PATH):
    os.makedirs(BURSTS_PATH)

# Get all directories that have valid pickle files

directories = get_immediate_subdirectories(os.path.dirname(os.path.abspath(__file__)))
packetDirs = [x for x in directories if "savedPackets" in x]

result = {}

#print(packetDirs)

for directory in packetDirs:
    
    # Get all filenames
    f = []
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)):
        f.extend(filenames)
        break

    for fileName in f:
        print("Processing: " + fileName)
        a = rdpcap(os.path.join(os.path.dirname(os.path.abspath(__file__)),  directory, fileName))

        nextPcap = []
        times = []

        burstNumber = 1

        currentTime = a[0].time

        for pkt in a:
            if (pkt.time - currentTime) < TIME_INTERVAL:
                nextPcap.append(pkt)
                currentTime = pkt.time
            else:
                wrpcap(os.path.join(BURSTS_PATH, fileName.split(".")[0] + "burst" + str(burstNumber) + ".pcap"), nextPcap)
                burstNumber += 1
                currentTime = pkt.time
                nextPcap = [pkt]

        wrpcap(os.path.join(BURSTS_PATH, fileName.split(".")[0] + "burst" + str(burstNumber) + ".pcap"), nextPcap)

