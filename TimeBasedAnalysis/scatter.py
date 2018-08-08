"""
Generates scatter plots comparing incoming/outgoing packet lengths with times
Also calculates and prints linear regression values to terminal
"""
import matplotlib.pyplot as plot
import pylab, csv, scipy.stats


lengthsTo = []
lengthsFrom = []
meanLenTo = []
meanLenFrom = []

meanFrom54 = []
meanTo54 = []

with open("featuresNo54.csv") as File:
    reader = csv.reader(File)
    firstRow = True
    for row in reader:
        if firstRow:
            firstRow = False
            continue
        lengthsTo.append(float(row[2]))
        lengthsFrom.append(float(row[3]))
        meanLenTo.append(float(row[6]))
        meanLenFrom.append(float(row[11]))
        meanTo54.append(float(row[14]))
        meanFrom54.append(float(row[15]))

fig = plot.figure()

ax = fig.add_subplot(111, frameon=False)

plot.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plot.grid(False)
ax.set_ylabel("Mean Packet Length (Bytes)")
ax.set_xlabel("Time for question/response (secs)")

ax1 = fig.add_subplot(221)
ax1.scatter(lengthsTo, meanLenTo)
ax1.set_title("Outgoing packets in bursts")

ax2 = fig.add_subplot(222)
ax2.scatter(lengthsFrom, meanLenFrom)
ax2.set_title("Incoming packets in bursts")

ax3 = fig.add_subplot(223)
ax3.scatter(lengthsTo, meanTo54)
ax3.set_title("Outgoing packets ignoring ACK")

ax4 = fig.add_subplot(224)
ax4.scatter(lengthsFrom, meanFrom54)
ax4.set_title("Incoming packets ignoring ACK")

print(scipy.stats.linregress(lengthsTo, meanLenTo))
print(scipy.stats.linregress(lengthsFrom, meanLenFrom))
print(scipy.stats.linregress(lengthsTo, meanTo54))
print(scipy.stats.linregress(lengthsFrom, meanFrom54))

"""
LinregressResult(slope=-34.504365447843966, intercept=556.83663931536773, rvalue=-0.5087230677153346, pvalue=0.044184048785573068, stderr=15.606177700978842)
LinregressResult(slope=35.311810685669229, intercept=193.13056450995515, rvalue=0.96296303499898539, pvalue=2.324044218751456e-09, stderr=2.642531114431327)
LinregressResult(slope=-39.800650520512967, intercept=686.59349150065145, rvalue=-0.7620355835278021, pvalue=0.00060101300089372017, stderr=9.0388722589958856)
LinregressResult(slope=33.915055176948997, intercept=777.11288524378642, rvalue=0.93053912563670216, pvalue=1.7386260135111136e-07, stderr=3.567001554805096)
"""

plot.show()