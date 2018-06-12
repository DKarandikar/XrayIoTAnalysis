"""
Get audio files, run to > 20 mins
While that occurs, capture traffic, then save as PCAP with name of audio
"""
import os, pygame, threading
from scapy.all import sniff
from subprocess import call

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
    fullPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audioFiles", filename)
    print(fullPath)
    call(["vlc", "--LZ", fullPath])

for file in getFiles():
    t = threading.Thread(target=loopSong, args=(file, ) , name="PlayMusic")
    t.start()
    #packets = sniff(timeout=1200, iface=INTERFACE_NAME)
