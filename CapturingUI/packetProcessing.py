import pyshark, pickle, os, json
import datetime

now = datetime.datetime.now()

date = "%d-%d-%d" % (now.day, now.month, now.year)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"config.json")) as f:
    data = json.load(f)
    HOME_NET_PREFIX = data["home_ip"]

PACKETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pickledPackets" + date)

if not os.path.exists(PACKETS_PATH):
    os.makedirs(PACKETS_PATH)

def processPacket(packet, ipDict):
    """
    Adds packet to ipDict based on the HOME_NET prefix
    Returns the modified dictionary
    """
    try:
        if HOME_NET_PREFIX in packet.ip.src:
            ipDict[packet.ip.src].append(packet)
        if HOME_NET_PREFIX in packet.ip.dst:
            ipDict[packet.ip.dst].append(packet)
    except AttributeError:
        # This can happen if a packet has no IP layer
        # We will just ignore packets like this
        print("Attribute Error")

    return(ipDict)

def savingPackets(IP, device, action, ipDict):
    """
    Saves packets in a .p file for the IP with device and Action Name
    Clears that entry from ipDict and returns it
    """
    counter = 0
    while True:
        if os.path.isfile(os.path.join(PACKETS_PATH, device + action + str(counter) + ".p")):
            counter += 1
        else:
            break
    pickle.dump(ipDict[IP], open(os.path.join(PACKETS_PATH, device + action + str(counter) + ".p"), "wb") )
    ipDict[IP] = []
    return(ipDict)