import pickle, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats

outgoing = True

ORIGINAL_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataProc", "data",  "FlowFeaturesTime.csv")

durations = np.genfromtxt(ORIGINAL_DATA, delimiter=",", usecols=2)

number = np.genfromtxt(ORIGINAL_DATA, delimiter=",", usecols=23)

category = np.genfromtxt(ORIGINAL_DATA, delimiter=",", usecols=1)

removedDur = []
removedNum = []

for index in range(len(durations)):
    if category[index] != 10 and category[index] != 11:
        removedDur.append(durations[index])
        removedNum.append(number[index])

print(scipy.stats.linregress(removedDur, removedNum))

if outgoing == False:
    durations = np.genfromtxt(ORIGINAL_DATA, delimiter=",", usecols=3)
    number = np.genfromtxt(ORIGINAL_DATA, delimiter=",", usecols=41)
    category = np.genfromtxt(ORIGINAL_DATA, delimiter=",", usecols=1)

data = []

for value in [i for i in range(1,12)]:
    this = []
    durs = []
    nums = []
    for index in range(len(durations)):
        if category[index] == value:
            durs.append(durations[index])
            nums.append(number[index])
    data.append((np.array(durs),np.array(nums)))
            
groups = [str(i) for i in range(1,12)]

colors = cm.rainbow(np.linspace(0, 1, len(groups)))


#x = np.arange(len(test)) 

for data, color, group in zip(tuple(data), tuple(colors), tuple(groups)):
    #print(data)
    x, y = data
    plt.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

    if outgoing:
        plt.xlabel("Duration of command (s)")
        plt.ylabel("Outgoing packet number")
    else:
        plt.xlabel("Duration of response (s)")
        plt.ylabel("Incoming packet number")

plt.legend(loc=4)

plt.show()