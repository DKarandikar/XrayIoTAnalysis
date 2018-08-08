"""
Test getting blocks of high aplitude out of a file
Example output:
Length is 60.79 seconds
Block finished from 0.82 to 2.58
Block finished from 15.41 to 17.11
Block finished from 30.07 to 31.81
Block finished from 36.51 to 37.01
Block finished from 45.79 to 47.48
"""
import os, copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))
fs, data = wavfile.read(os.path.join(FILE_PATH, "audioFilesWav", "Unknown - AlexaTime1Rec.wav"))
ALLOWED_DIP_FRAMES = fs

print("Length is %.2f seconds" % (len(data)*1.0 / float(fs)))

highest = np.max(np.abs(data))

highCutoff = highest / 5

counter = 0
low = True
started = 0
highLowValues=0

cutoffs = []

while counter < len(data):
    
    if np.abs(data[counter,0]) < highCutoff and low:
        pass
    elif np.abs(data[counter,0]) < highCutoff and not low:
        highLowValues += 1
        if highLowValues > ALLOWED_DIP_FRAMES:
            print("Block finished from %.2f to %.2f" % (started*1.0/float(fs) , (counter-ALLOWED_DIP_FRAMES/2)*1.0/float(fs)))
            cutoffs.append((started, (counter-int(ALLOWED_DIP_FRAMES/2))))
            low = not low
            highLowValues = 0
        
    elif np.abs(data[counter,0]) >= highCutoff and low:
        started = copy.deepcopy(counter)
        highLowValues = 0
        low = not low
    elif np.abs(data[counter,0]) >= highCutoff and not low:
        highLowValues = 0


    counter += 1

xvals = np.arange(len(data))

plt.plot(xvals, data[:,0])

#plt.show()

# Can't use enumerate, might skip some
tick = 0

for values in cutoffs:
    # Ensure they're at least 1 second
    if values[1] - values[0] > fs:
        wavfile.write(os.path.join(FILE_PATH, "audioCutUps", "AlexaTime1Rec" + str(tick) + ".wav"), fs, data[values[0]:values[1],:])
        tick += 1