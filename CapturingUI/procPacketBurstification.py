import os, pyshark, pickle, inspect, json

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

BURSTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bursts")
TIME_INTERVAL = 1.0

if not os.path.exists(BURSTS_PATH):
    os.makedirs(BURSTS_PATH)

# Get all directories that have valid pickle files

directories = get_immediate_subdirectories(os.path.dirname(os.path.abspath(__file__)))
packetDirs = [x for x in directories if "pickledPackets" in x]

result = {}

print(packetDirs)

for directory in packetDirs:
    
    # Get all filenames
    f = []
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)):
        f.extend(filenames)
        break

    for pickledFile in f:
        print(pickledFile)
        packets = pickle.load(open( os.path.join(os.path.dirname(os.path.abspath(__file__)),  directory, pickledFile), "rb"))

        is_summary = False
        try:
            # This only works when it isn't a summary
            temp = packets[0].ip.src
        except AttributeError:
            # Hence if it doesn't work, it's a summary
            is_summary = True

        # Now we can just extract bursts
        if is_summary:
            currentTime = float(packets[0].time)
        else:
            currentTime = float(packets[0].sniff_timestamp)

        nextBurst = []
        burstNumber = 0

        # Iterate through packets and seperate a burst when one is more than TIME_INTERVAL later than the previous
        for pkt in packets:
            if is_summary:
                packetTime = float(pkt.time)
            else:
                packetTime = float(pkt.sniff_timestamp)

            if (packetTime - currentTime) < TIME_INTERVAL:
                nextBurst.append(pkt)
                if is_summary:
                    currentTime = float(pkt.time)
                else:
                    currentTime = float(pkt.sniff_timestamp)
            else:
                filename = directory.split("Packets")[1] + "--" + pickledFile.split(".")[0] + "burst" + str(burstNumber) + ".p"
                pickle.dump(nextBurst, open(os.path.join(BURSTS_PATH, filename), "wb"))
                
                burstNumber += 1
                if is_summary:
                    currentTime = float(pkt.time)
                else:
                    currentTime = float(pkt.sniff_timestamp)
                nextPcap = [pkt]

        filename = directory.split("Packets")[1] + "--" + pickledFile.split(".")[0] + "burst" + str(burstNumber) + ".p"
        pickle.dump(nextBurst, open(os.path.join(BURSTS_PATH, filename), "wb"))

