import pyaudio, struct, math, datetime, os, threading, time, copy
import numpy as np
from scapy.all import sniff, wrpcap
from subprocess import call, Popen, PIPE, getoutput
from multiprocessing import Pool

import statisticProcessing

"""
Repeat:
**
Setup audio and packet sniffing
Start sniffing
Run audio
Start recording
End rec, process
End sniff, process
**

"""

INTERFACE_NAME = "wlan0"
DEVICE_IP = "192.168.4.2"

SHORT_NORMALIZE = (1.0/32768.0)
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
EXTRA_SNIFF_SECONDS = 5

ALLOWED_DIP_FRAMES = 4

FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))

def getFileLength(filename):
    mycmd = getoutput("ffmpeg -i " + filename + " 2>&1 | grep Duration | cut -d ' ' -f 4 | sed s/,// ")
    return mycmd

def getFiles():
    """ Gets all audio files in a list"""
    mypath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audioCutUps")
    files = []
    for (_, _, filenames) in os.walk(mypath):
        files.extend(filenames)
        break
    return files

def get_rms( block ):
    """ c.f. https://stackoverflow.com/questions/36413567/pyaudio-convert-stream-read-into-int-to-get-amplitude#36413872 """
    # RMS amplitude is defined as the square root of the 
    # mean over time of the square of the amplitude.
    # so we need to convert this string of bytes into 
    # a string of 16-bit samples...

    # we will get one short out for each 
    # two chars in the string.
    count = len(block)/2
    format = "%dh"%(count)
    shorts = struct.unpack( format, block )

    # iterate over the block.
    sum_squares = 0.0
    for sample in shorts:
        # sample is a signed short in +/- 32768. 
        # normalize it to 1.0
        n = sample * SHORT_NORMALIZE
        sum_squares += n*n

    return math.sqrt( sum_squares / count )

def getCutoffs(listInts):
    cutoffs = []
    highest = np.max(np.abs(listInts))

    print(len(listInts))

    highCutoff = highest / 5

    counter = 0
    low = True
    started = 0
    highLowValues=0

    while counter < len(listInts):
        
        if np.abs(listInts[counter]) < highCutoff and low:
            pass
        elif np.abs(listInts[counter]) < highCutoff and not low:
            highLowValues += 1
            if highLowValues > ALLOWED_DIP_FRAMES:
                print("Block finished from %.2f to %.2f" % (started*1.0*CHUNK/float(RATE) , (counter-ALLOWED_DIP_FRAMES/2)*1.0*CHUNK/float(RATE)))
                cutoffs.append((started*1.0*CHUNK/float(RATE), (counter-ALLOWED_DIP_FRAMES/2)*1.0*CHUNK/float(RATE)))
                low = not low
                highLowValues = 0
            
        elif np.abs(listInts[counter]) >= highCutoff and low:
            started = copy.deepcopy(counter)
            highLowValues = 0
            low = not low
        elif np.abs(listInts[counter]) >= highCutoff and not low:
            highLowValues = 0


        counter += 1
    
    return cutoffs

def sniffPackets(time):

    global result

    print(time + RECORD_SECONDS + EXTRA_SNIFF_SECONDS)
    
    packets = sniff(filter="ip " + DEVICE_IP , timeout=time + RECORD_SECONDS + EXTRA_SNIFF_SECONDS, iface=INTERFACE_NAME)

    result = packets

    #print(result)
    
def convertFloatLength(string):
    hours = string.split(":")[0]
    minutes = string.split(":")[1]
    seconds = string.split(":")[2].split(".")[0]
    return (3600*int(hours) + 60*int(minutes) + int(seconds) + 1)

def getExactLength(string):
    hours = string.split(":")[0]
    minutes = string.split(":")[1]
    seconds = string.split(":")[2]
    return (float(3600*int(hours) + 60*int(minutes) + float(seconds)))

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

def saveStatistics(listListListFloats, filename, duration, response):
    
    now = datetime.datetime.now()
    date = "%d-%d-%d" % (now.day, now.month, now.year)
    packetsPath = os.path.join(FILE_PATH, "savedPackets" + date)
    counter = 0
    while True:
        if os.path.isfile(os.path.join(packetsPath, filename + str(counter) + ".pcap")):
            counter += 1
        else:
            break


    for bNo, listListFloats in enumerate(listListListFloats):

        counter -=1

        fCounter = 0
        for listFloats in listListFloats:
            row = []
            row.append(filename + str(counter) + "On" + date + "Burst" + str(bNo) + "Flow" + str(fCounter))

            classNumber = statisticProcessing.getFlowClass(filename)

            row.append(classNumber)

            row.append(duration)

            row.append(response)

            row.extend(listFloats)

            writer = statisticProcessing.getCSVWriter(True)

            writer.writerow(row)

            fCounter += 1

for file in getFiles():

    p = pyaudio.PyAudio()

    # Sniff here
    global result

    duration = convertFloatLength(getFileLength(os.path.join(FILE_PATH, "audioCutUps",file)))

    t = threading.Thread(target=sniffPackets, args=(duration, ) , name="SniffTraffic")
    t.start()

    # Play audio

    playlist = [os.path.join(FILE_PATH, "audioCutUps", file)]
    command = ["cvlc", "--play-and-exit"]
    command.extend(playlist)
    call(command)

    # Record

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(get_rms(data))

    stream.stop_stream()
    stream.close()
        
    # Wait for sniffing to be done
    t.join()

    # Process

    cutoffs = getCutoffs(frames)
    responseTimes = []
    for val in cutoffs:
        responseTimes.append(val[1]-val[0])
    print(cutoffs)
    longestResponse = np.max(responseTimes)

    savePackets(result, file.split(".")[0])

    statistics = statisticProcessing.processPackets(result, file.split(".")[0], True)

    preciseDuration = getExactLength(getFileLength(os.path.join(FILE_PATH, "audioCutUps",file)))

    print(preciseDuration)

    saveStatistics(statistics, file.split(".")[0], preciseDuration, longestResponse)


#print(frames)
