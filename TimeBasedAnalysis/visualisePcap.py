"""
Script for visualising a pcap file
Displays a graph with time on x-axis and packet length on the y-axis 
"""
import statistics, pyshark, os
import matplotlib.pyplot as plt

pkts = pyshark.FileCapture(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pcaps", "NokiaScaleSingleUse0.pcap"))

DEVICE_IP = "192.168.4.12"
PHONE_IP = "192.168.4.19"

IGNORE_PHONE = False    # Whether or not to ignore packets to/from the phone IP 
TIME_INTERVAL = False   # Whether or not to restrict to only a set time interval

START_TIME = 79
END_TIME = 124

initialTime = float(pkts[0].sniff_timestamp)

#print(initialTime)

inTimes = []
inSize = []

outTimes = []
outSize = []

for p in pkts:
    if 'IP' in p:
        #print(p.sniff_timestamp)
        try:
            if (IGNORE_PHONE and p['ip'].src != PHONE_IP and p['ip'].dst != PHONE_IP) or not IGNORE_PHONE:
                #print(float(p.sniff_timestamp) - initialTime)
                if (TIME_INTERVAL and float(p.sniff_timestamp) - initialTime < END_TIME and float(p.sniff_timestamp) - initialTime > START_TIME) or not TIME_INTERVAL:
                    if p['ip'].src == DEVICE_IP:
                        outTimes.append(float(p.sniff_timestamp) - initialTime)
                        outSize.append(int(p.length))
                    else:
                        inTimes.append(float(p.sniff_timestamp) - initialTime)
                        inSize.append(int(p.length))
        except AttributeError:
            print("Attribute Error")

fig, ax = plt.subplots()

fig.suptitle('Incoming (blue) and outgoing (red) packet sizes')
plt.ylabel('Packet size (Bytes)')
plt.xlabel('Time (secs)')

ax.stem(inTimes, inSize, 'b', markerfmt=' ', basefmt=" ")
ax.stem(outTimes, outSize, 'r', markerfmt=' ', basefmt=" ")
plt.show()