import os, csv, random

if os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "randomised.csv")):
    os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "randomised.csv"))

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "randomised.csv"),'a') as cleanFile:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "normalized.csv"),"r") as feat:
        features = csv.reader(feat)
        randomed = csv.writer(cleanFile)

        for row in features:
            if row:
                row2 = row
                row2[1] = random.randint(1,8)
                randomed.writerow(row2)
