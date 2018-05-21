import matplotlib.pyplot as plot
import pylab, csv, scipy.stats


lengthsTo = []
lengthsFrom = []

times = []

with open("featuresBurstTime.csv") as File:
    reader = csv.reader(File)
    firstRow = True
    for row in reader:
        if firstRow:
            firstRow = False
            continue
        lengthsTo.append(float(row[2]))
        lengthsFrom.append(float(row[3]))
        times.append(float(row[16]))


fig = plot.figure()

#plot.scatter([sum(x) for x in zip(lengthsTo, lengthsFrom)], times)
#plot.scatter(lengthsFrom, times)
#plot.scatter([b-a for a,b in zip(lengthsTo, lengthsFrom)], times)

plot.scatter(lengthsFrom, [a-b for a, b in zip(times, lengthsTo)])

plot.show()

for x in range(len(times)):
    print(times[x] - lengthsTo[x])