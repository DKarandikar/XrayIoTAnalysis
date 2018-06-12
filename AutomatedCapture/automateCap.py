"""
Get audio files, run to > 20 mins
While that occurs, capture traffic, then save as PCAP with name of audio
"""
import os, pygame, threading
from scapy.all import sniff
from subprocess import call, Popen, PIPE

INTERFACE_NAME = "wlan0"
DEVICE_IP = "192.168.4.2"

def getFiles():
    """ Gets all audio files in a list"""
    mypath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audioFiles")
    files = []
    for (_, _, filenames) in os.walk(mypath):
        files.extend(filenames)
        break
    return files

def loopSong(filename):
    
    call(["cvlc", filename])

for file in getFiles():
    fullPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audioFiles", file)

    p1 = Popen(["ffmpeg", "-i", fullPath, "2>&l"], stdout=PIPE)
    p2 = Popen(["grep", "duration"], stdin=p1.stdout, stdout=PIPE)
    p3 = Popen(["cut", "-d", r"""' '""", "-f", "4"], stdin=p2.stdout, stdout=PIPE)
    p4 = Popen(["sed", r"s/,//"], stdin=p3.stdout, stdout=PIPE)

    output = p4.communicate()[0]

    print(output)

    t = threading.Thread(target=loopSong, args=(fullPath, ) , name="PlayMusic")
    t.start()
    packets = sniff(timeout=1200, iface=INTERFACE_NAME)
