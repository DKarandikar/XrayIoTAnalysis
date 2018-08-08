"""
Got all IPs out of pickled packets, but we don't use this anymore
"""
import os, inspect, json
from scapy.all import rdpcap, IP

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

# Get all directories that have valid pickle files

directories = get_immediate_subdirectories(os.path.dirname(os.path.abspath(__file__)))
packetDirs = [x for x in directories if "savedPackets" in x]

result = {}

for directory in packetDirs:
    
    # Get all filenames
    f = []
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)):
        f.extend(filenames)
        break

    for fileName in f:
        packets = rdpcap(os.path.join(os.path.dirname(os.path.abspath(__file__)),  directory, fileName))

        # Now we can just extract the IPs 

        ips = set()
        for packet in packets:
            ips.add(packet[IP].src)
            ips.add(packet[IP].dst)

        key = directory.split("Packets")[1] + "--" +  fileName.split(".")[0]

        result[key] = sorted(list(ips))

print(result)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ips.json"), "w") as fp:
    json.dump(result, fp, sort_keys=True, indent=4, separators=(',', ': '))