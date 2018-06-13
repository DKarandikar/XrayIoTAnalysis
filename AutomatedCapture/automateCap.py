"""
Get audio files, run to > 20 mins
While that occurs, capture traffic, then save as PCAP with name of audio
"""
import os, threading, math
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

def loopSong(filename, plays):
    for x in range(plays):
        call(["cvlc", filename])

def getFileLength(filename):
    p1 = Popen(["ffmpeg", "-i", filename, "2>&l"], stdout=PIPE)
    p2 = Popen(["grep", "duration"], stdin=p1.stdout, stdout=PIPE)
    p3 = Popen(["cut", "-d", r"""' '""", "-f", "4"], stdin=p2.stdout, stdout=PIPE)
    p4 = Popen(["sed", r"s/,//"], stdin=p3.stdout, stdout=PIPE)
    return p4.communicate()[0]

def convertFloatLength(string):
    hours = string.split(":")[0]
    minutes = string.split(":")[1]
    seconds = string.split(":")[2].split(".")[0]

    return (3600*hours + 60*minutes + seconds + 1)

for file in getFiles():
    fullPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audioFiles", file)

    fileLength = getFileLength(fullPath)

    lengthFloat = convertFloatLength(fileLength)

    numberPlays = math.floor(1.0* 1200 / float(lengthFloat))

    t = threading.Thread(target=loopSong, args=(fullPath, numberPlays ) , name="PlayMusic")
    t.start()
    packets = sniff(timeout=1200, iface=INTERFACE_NAME)
