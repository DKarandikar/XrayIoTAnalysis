import pickle, os, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats

outgoing = False
binSameDuration = True

ROUNDING_DP = 2

ORIGINAL_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataProc", "data",  "FlowFeaturesTime (1).csv")

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

if binSameDuration:
    
    tupDat = []
    for index in range(len(durations)):
        tupDat.append((round(durations[index],ROUNDING_DP), number[index], category[index]))

    tupDat = sorted(tupDat, key=lambda tup: (tup[0],tup[2]) )

    #print(tupDat)

    durations = []
    number = []
    category = []

    for index in range(len(tupDat)):
        durations.append(tupDat[index][0])
        number.append(tupDat[index][1])
        category.append(tupDat[index][2])

    #print(durations)

    newNum = []
    newErr = []
    newDur = []
    newCat = []

    index = 0

    while index < len(durations) - 1:
        thisDur = durations[index]
        thisCat = category[index]
        theseNum = [number[index]]
        try:
            while durations[index + 1] == thisDur and category[index + 1] == thisCat:
                if number[index+1] > 50:
                    theseNum.append(number[index + 1])
                index += 1
        except IndexError:
            pass

        newDur.append(thisDur)
        newCat.append(thisCat)
        newNum.append(np.mean(theseNum))
        newErr.append(np.std(theseNum))
        index += 1

    #print(newDur)
    #print(newCat)

    data = []

    for value in [i for i in range(1,12)]:
        this = []
        durs = []
        nums = []
        errs = []
        for index in range(len(newDur)):
            if int(newCat[index]) == value:
                durs.append(newDur[index])
                nums.append(newNum[index])
                errs.append(newErr[index])
        data.append((np.array(durs),np.array(nums), np.array(errs)))

    #print(data)

    for index, p in enumerate(data):
        x,y,z = p
        if len(x) != 0:
            print(index+1, scipy.stats.linregress(x, y).rvalue)

    

else:
    

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

if binSameDuration:
    
    for data, color, group in zip(tuple(data), tuple(colors), tuple(groups)):
        #print(data)
        x, y, z = data
        plt.errorbar(x, y, alpha=0.8, c=color, yerr=z, label=group, fmt='o')

        if outgoing:
            plt.xlabel("Duration of command (s)")
            plt.ylabel("Outgoing packet number")
        else:
            plt.xlabel("Duration of response (s)")
            plt.ylabel("Incoming packet number")

    #plt.legend(loc=4)

    plt.show()
    
else:

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