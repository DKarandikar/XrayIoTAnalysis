import os, pickle

# Not used anymore, pickling the packets was a bad idea
# Scapy was far better than pyshark in this instance 

PCAPS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pickledPackets30-5-2018")

f = []
for (dirpath, dirnames, filenames) in os.walk(PCAPS_PATH):
    f.extend(filenames)
    break

for file in f:
    k = pickle.load(open(os.path.join(PCAPS_PATH, file), "rb"))
    print(len(k))

