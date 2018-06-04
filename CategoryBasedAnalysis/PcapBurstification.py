from scapy.all import *

PCAPS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pcaps")
BURSTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bursts")

try:
        a = rdpcap(os.path.join(PCAPS_PATH, sys.argv[1]))
except:
        print("Please give a filename as argument")

nextPcap = []
times = []

timeInterval = 1.0
burstNumber = 1

currentTime = a[0].time

name = sys.argv[1]

if "." in name:
        name = name.split(".")[0]

for pkt in a:
        if (pkt.time - currentTime) < timeInterval:
                nextPcap.append(pkt)
                currentTime = pkt.time
        else:
                wrpcap(os.path.join(BURSTS_PATH, name + "burst" + str(burstNumber) + ".pcap"), nextPcap)
                burstNumber += 1
                currentTime = pkt.time
                nextPcap = [pkt]

wrpcap(os.path.join(BURSTS_PATH, name + "burst" + str(burstNumber) + ".pcap"), nextPcap)
