import pyshark, pickle, os, json, threading
import datetime

from scapy.all import IP, wrpcap

now = datetime.datetime.now()

date = "%d-%d-%d" % (now.day, now.month, now.year)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"config.json")) as f:
    data = json.load(f)
    HOME_NET_PREFIX = data["home_ip"]
    summaries = data["only_summaries"] == "True"

PACKETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "savedPackets" + date)

if not os.path.exists(PACKETS_PATH):
    os.makedirs(PACKETS_PATH)

def processPacket(packet, ipDict):
    """
    Adds packet to ipDict based on the HOME_NET prefix
    Returns the modified dictionary
    """
    try:
        if HOME_NET_PREFIX in packet[IP].src:
            ipDict[packet[IP].src].append(packet)
        if HOME_NET_PREFIX in packet[IP].dst:
            ipDict[packet[IP].dst].append(packet)

    except IndexError:
        # This can happen if a packet has no IP layer
        # We will just ignore packets like this
        print("Attribute Error")

    return(ipDict)

def savingPackets(IP, device, action, ipDict, label):
    """
    Saves packets in a .p file for the IP with device and Action Name
    Clears that entry from ipDict and returns it
    """
    counter = 0
    while True:
        if os.path.isfile(os.path.join(PACKETS_PATH, device + action + str(counter) + ".pcap")):
            counter += 1
        else:
            break

    # Create a thread to pickle the list because it's a bit slow
    # Use list(ipDict[IP].) to copy the list so that modifying it later isn't a problem
    # str(counter) also serves a similar purpose (unintentionally but happily)
    t = threading.Thread( target=saveThread, args=(list(ipDict[IP]), device + action + str(counter) + ".pcap", label) , name="Saving")
    t.start()

    ipDict[IP] = []
    return(ipDict)

def saveThread(listToSave, fileName, label):
    if len(listToSave) > 0:
        wrpcap(os.path.join(PACKETS_PATH, fileName), listToSave)
        label.config(text="Saved " + fileName)
    else:
        label.config(test="Nothing to save")
