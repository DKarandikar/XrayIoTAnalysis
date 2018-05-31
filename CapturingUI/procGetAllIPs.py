import os, pyshark, pickle, inspect, json

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

# Get all directories that have valid pickle files

directories = get_immediate_subdirectories(os.path.dirname(os.path.abspath(__file__)))
packetDirs = [x for x in directories if "pickledPackets" in x]

result = {}

for directory in packetDirs:
    
    # Get all filenames
    f = []
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)):
        f.extend(filenames)
        break

    for pickledFile in f:
        packets = pickle.load(open( os.path.join(os.path.dirname(os.path.abspath(__file__)),  directory, pickledFile), "rb"))

        is_summary = False
        try:
            # This only works when it isn't a summary
            temp = packets[0].ip.src
        except AttributeError:
            # Hence if it doesn't work, it's a summary
            is_summary = True

        # Now we can just extract the IPs 

        ips = set()
        if is_summary:
            for packet in packets:
                ips.add(packet.source)
                ips.add(packet.destination)
        else:
            for packet in packets:
                ips.add(packet.ip.src)
                ips.add(packet.ip.dst)

        key = directory.split("Packets")[1] + "--" +  pickledFile.split(".")[0]

        result[key] = sorted(list(ips))

with open("ips.json", "w") as fp:
    json.dump(result, fp, sort_keys=True, indent=4, separators=(',', ': '))