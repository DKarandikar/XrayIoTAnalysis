import pyaudio, struct, math, datetime, os, threading, time
import numpy as np
from scapy.all import sniff, wrpcap
from subprocess import call, Popen, PIPE, getoutput
from multiprocessing import Pool

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
RECORD_SECONDS = 10

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
                print("Block finished from %.2f to %.2f" % (started*1.0/float(fs) , (counter-ALLOWED_DIP_FRAMES/2)*1.0/float(fs)))
                cutoffs.append((started, (counter-int(ALLOWED_DIP_FRAMES/2))))
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

    print(time + RECORD_SECONDS + 10)
    
    packets = sniff(filter="ip " + DEVICE_IP , timeout=time + RECORD_SECONDS + 10, iface=INTERFACE_NAME)

    result = packets

    print(result)
    
def convertFloatLength(string):
    hours = string.split(":")[0]
    minutes = string.split(":")[1]
    seconds = string.split(":")[2].split(".")[0]
    return (3600*int(hours) + 60*int(minutes) + int(seconds) + 1)

for file in getFiles():

    p = pyaudio.PyAudio()

    # Sniff here


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
    
    print(frames)


#print(frames)
