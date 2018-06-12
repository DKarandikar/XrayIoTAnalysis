import os, csv, random

if os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "onlyIncoming.csv")):
    os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "onlyIncoming.csv"))

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "onlyIncoming.csv"),'a') as cleanFile:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "normalized.csv"),"r") as feat:
        features = csv.reader(feat)
        randomed = csv.writer(cleanFile)

        for row in features:
            if row:
                row2 = row[0:2] + row[20:38]
                randomed.writerow(row2)
