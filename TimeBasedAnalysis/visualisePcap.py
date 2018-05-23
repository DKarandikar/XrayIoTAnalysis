import statistics, pyshark, os
import matplotlib.pyplot as plt

pkts = pyshark.FileCapture(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pcaps", "HueOnOff1.pcap"))

DEVICE_IP = "192.168.4.16"
PHONE_IP = "192.168.4.19"

IGNORE_PHONE = True

initialTime = float(pkts[0].sniff_timestamp)

inTimes = []
inSize = []

outTimes = []
outSize = []

for p in pkts:
    if 'IP' in p:
        #print(p.sniff_timestamp)
        try:
            if (IGNORE_PHONE and p['ip'].src != PHONE_IP and p['ip'].dst != PHONE_IP) or not IGNORE_PHONE:
                if p['ip'].src == DEVICE_IP:
                    outTimes.append(float(p.sniff_timestamp) - initialTime)
                    outSize.append(int(p.length))
                else:
                    inTimes.append(float(p.sniff_timestamp) - initialTime)
                    inSize.append(int(p.length))
        except AttributeError:
            print("Attribute Error")

fig, ax = plt.subplots()

fig.suptitle('Incoming (blue) and outgoing (red) packet sizes during a burst')
plt.ylabel('Packet size (Bytes)')
plt.xlabel('Time (secs)')

ax.stem(inTimes, inSize, 'b', markerfmt=' ', basefmt=" ")
ax.stem(outTimes, outSize, 'r', markerfmt=' ', basefmt=" ")
plt.show()