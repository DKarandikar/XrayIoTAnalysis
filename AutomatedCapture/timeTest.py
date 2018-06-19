import pyaudio, struct, math, datetime, os, threading
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import numpy as np
from scapy.all import sniff, wrpcap
from subprocess import call, Popen, PIPE, getoutput

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

SHORT_NORMALIZE = (1.0/32768.0)
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10

FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))

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

p = pyaudio.PyAudio()



# Sniff here

# Play audio

playlist = [os.path.join(FILE_PATH, "audioCutUps", "AlexaTime1Rec0.wav")]
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

print(frames)


xvals = np.arange(len(frames))

plt.plot(xvals, frames)

plt.show()
