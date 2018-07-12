"""
Get audio files, run to > 20 mins
While that occurs, capture traffic, then save as PCAP with name of audio
"""
import os, threading, math, datetime, gc
from scapy.all import sniff, wrpcap
from subprocess import call, Popen, PIPE, getoutput

import statisticProcessing

INTERFACE_NAME = "wlan0"
DEVICE_IP = "192.168.4.2"
SESSION_LENGTH = 600 # Max seconds to play, 1200 for Alexa is ok, 600 better for Google Home 
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))

FILE_LENGTHS = {'GoogleLightsBrightDim30sec.m4a': "00:00:30.99",
                'GoogleShoppingList1min.m4a': "00:01:01.99",
                'GoogleJoke1Min.m4a': "00:01:01.99",
                'GoogleTime1Min.m4a': "00:01:01.99",
                'GoogleAlarms1min.m4a': "00:01:01.99",
                'GoogleLightsOnOff30sec.m4a': "00:00:30.99"}


def getFiles():
    """ Gets all audio files in a list"""
    mypath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audioFiles")
    files = []
    for (_, _, filenames) in os.walk(mypath):
        files.extend(filenames)
        break
    return files

def loopSong(filename, plays):
    playlist = []
    for x in range(plays):
        playlist.append(filename)

    command = ["cvlc", "--play-and-exit"]
    command.extend(playlist)
    call(command)

def getFileLength(filename):
    if filename in FILE_LENGTHS.keys():
        return FILE_LENGTHS[filename]
    else:
        mycmd = getoutput("ffmpeg -i " + filename + " 2>&1 | grep Duration | cut -d ' ' -f 4 | sed s/,// ")
        FILE_LENGTHS[filename] = mycmd
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

def playAndSave(fullPath, file):
    """ Play full path, capture, process and save """
    time = playForLessXMins(fullPath)

    print("Capturing for " + str(time + 15) + " seconds")

    packets = sniff(filter="ip " + DEVICE_IP , timeout=time + 15, iface=INTERFACE_NAME)

    savePackets(packets, file.split(".")[0])

    statisticProcessing.processPackets(packets, file.split(".")[0])

def main():
    try:
        #print(getFiles())
        while True:
            for file in getFiles():
                
                fullPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audioFiles", file)

                playAndSave(fullPath, file)

                gc.collect()

    except KeyboardInterrupt:
        print("Interrupted")


if __name__ == '__main__':
    main()  
