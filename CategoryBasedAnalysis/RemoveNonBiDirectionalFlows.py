import os, csv

if os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "noNonBiDirect.csv")):
    os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "noNonBiDirect.csv"))

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "noNonBiDirect.csv"),'a') as cleanFile:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "flowFeatures.csv"),"r") as feat:
        features = csv.reader(feat)
        clean = csv.writer(cleanFile)

        for row in features:
            if row:
                if row[21] != "nan":
                    clean.writerow(row)
