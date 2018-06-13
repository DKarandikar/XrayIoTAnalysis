"""
Get audio files, run to > 20 mins
While that occurs, capture traffic, then save as PCAP with name of audio
"""
import os, threading, math, datetime
from scapy.all import sniff, wrpcap
from subprocess import call, Popen, PIPE, getoutput

INTERFACE_NAME = "wlan0"
DEVICE_IP = "192.168.4.2"
SESSION_LENGTH = 100 # Max seconds to play
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))


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
    mycmd = getoutput("ffmpeg -i " + filename + " 2>&1 | grep Duration | cut -d ' ' -f 4 | sed s/,// ")
    return mycmd

def convertFloatLength(string):
    hours = string.split(":")[0]
    minutes = string.split(":")[1]
    seconds = string.split(":")[2].split(".")[0]
    return (3600*int(hours) + 60*int(minutes) + int(seconds) + 1)

def playForLessXMins(filename):
    fileLength = getFileLength(filename)
    lengthFloat = convertFloatLength(fileLength)
    numberPlays = math.floor(1.0* SESSION_LENGTH / float(lengthFloat))
    print("File length is " + str(int(lengthFloat)) + " seconds, playing " + str(numberPlays) + " times" )
    t = threading.Thread(target=loopSong, args=(filename, numberPlays ) , name="PlayMusic")
    t.start()

    return int(math.floor(numberPlays*lengthFloat))
    
def savePackets(packets, filename):
    # Setup the folder
    now = datetime.datetime.now()
    date = "%d-%d-%d" % (now.day, now.month, now.year)
    packetsPath = os.path.join(FILE_PATH, "savedPackets" + date)
    if not os.path.exists(packetsPath):
        os.makedirs(packetsPath)

    # Get the number of this file
    counter = 0
    while True:
        if os.path.isfile(os.path.join(packetsPath, filename + str(counter) + ".pcap")):
            counter += 1
        else:
            break

    # Save the file
    if len(packets) > 0:
        wrpcap(os.path.join(packetsPath, filename + str(counter) + ".pcap"), packets)
        print("Saved file: " + filename + str(counter) + ".pcap")


    

def main():
    while True:
        for file in getFiles():
            
            fullPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audioFiles", file)

            time = playForLessXMins(fullPath)

            print("Capturing for " + str(time + 15) + " seconds")

            packets = sniff(filter="ip " + DEVICE_IP , timeout=time + 15, iface=INTERFACE_NAME, prn=lambda pkt: pkt.summary())

            savePackets(packets, file.split(".")[0])

            #processPackets(packets)


if __name__ == '__main__':
    main()  
