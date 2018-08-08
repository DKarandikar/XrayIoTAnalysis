"""
Script to display question/answer duration against number of packets in that burst
Option to bin the data or not, binned data has error bars for 1 std 
"""

import pickle, os, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats

outgoing = True
binSameDuration = True  # Bin categories?


ROUNDING_DP = 2

ORIGINAL_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataProc", "data",  "FlowFeaturesTime (1).csv")

durations = np.genfromtxt(ORIGINAL_DATA, delimiter=",", usecols=2)
number = np.genfromtxt(ORIGINAL_DATA, delimiter=",", usecols=23)
category = np.genfromtxt(ORIGINAL_DATA, delimiter=",", usecols=1)

if outgoing == False:
    durations = np.genfromtxt(ORIGINAL_DATA, delimiter=",", usecols=3)
    number = np.genfromtxt(ORIGINAL_DATA, delimiter=",", usecols=41)
    category = np.genfromtxt(ORIGINAL_DATA, delimiter=",", usecols=1)

if binSameDuration:
    
    # First sort by duration then by category
    tupDat = []
    for index in range(len(durations)):
        tupDat.append((round(durations[index],ROUNDING_DP), number[index], category[index]))

    tupDat = sorted(tupDat, key=lambda tup: (tup[0],tup[2]) )


    # Next setup the lists again (extract from tuples)

    durations = []
    number = []
    category = []

    for index in range(len(tupDat)):
        durations.append(tupDat[index][0])
        number.append(tupDat[index][1])
        category.append(tupDat[index][2])

    # Setup new lists

    newNum = []
    newErr = []
    newDur = []
    newCat = []

    index = 0

    # Go through the entries in order 

    while index < len(durations) - 1:
        thisDur = durations[index]
        thisCat = category[index]
        theseNum = [number[index]]

        # If more than one matches on category and duration add to list 

        try:
            while durations[index + 1] == thisDur and category[index + 1] == thisCat:
                if number[index+1] > 50:
                    theseNum.append(number[index + 1])
                index += 1
        except IndexError:
            pass

        newDur.append(thisDur)
        newCat.append(thisCat)
        # Get means and std of these lists 
        newNum.append(np.mean(theseNum))
        newErr.append(np.std(theseNum))
        index += 1

    #print(newDur)
    #print(newCat)

    data = []

    # Sort into numpy arrays in order of category 

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

    #Print pearson correlation coefficients for each category 

    for index, p in enumerate(data):
        x,y,z = p
        if len(x) != 0:
            print(index+1, scipy.stats.linregress(x, y).rvalue)

    

else:
    
    # Just sort into numpy arrays on category basis 

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
            
# Get groups and colours

groups = [str(i) for i in range(1,12)]

colors = cm.rainbow(np.linspace(0, 1, len(groups))) 


#x = np.arange(len(test)) 

if binSameDuration:
    
    # If binned use errorbar plots
    
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
    
    # If not binned use scatter plots

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