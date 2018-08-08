"""
Used when using live classify with min/max scaled predictions
Saves data that has been min/max scaled 
Not used anymore 
"""
import os, csv, random

if os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "liveCapFiles",  "minMax.csv")):
    os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), "liveCapFiles",  "minMax.csv"))

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "liveCapFiles",  "minMax.csv"),'a', newline='') as cleanFile:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataProc", data",  "FlowFeatures.csv"),"r") as feat:
        features = csv.reader(feat)
        minMax = csv.writer(cleanFile)

        minimum = []
        maximum = []

        for row in features:
            if row:
                row = [float(e) for e in row[2:]]

                if minimum:
                    minimum = [min(x,y) for x,y in zip(minimum, row)]
                    maximum = [max(x,y) for x,y in zip(maximum, row)]
                else:
                    minimum = row
                    maximum = row


        minMax.writerow(maximum)
        minMax.writerow(minimum)


print(minimum)
print(maximum)
