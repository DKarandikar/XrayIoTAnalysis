import os, csv

if os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "noNonBiDirectTwo.csv")):
    os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "noNonBiDirectTwo.csv"))

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "noNonBiDirectTwo.csv"),'a') as cleanFile:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "noNonBiDirect.csv"),"r") as feat:
        features = csv.reader(feat)
        clean = csv.writer(cleanFile)

        for row in features:
            if row:
                if row[1] == "1" or row[1] == "2":
                    clean.writerow(row)
