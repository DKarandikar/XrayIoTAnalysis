import os, csv

numbers = [0 for x in range(10)]
numbers2 = [0 for x in range(10)]

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "flowFeatures.csv"),"r") as feat:
    features = csv.reader(feat)
    for row in features:
        if row:
            numbers[int(row[1])] += 1

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "noNonBiDirect.csv"),"r") as feat:
    features = csv.reader(feat)
    for row in features:
        if row:
            numbers2[int(row[1])] += 1
print()
print( "Noise, time, weather, joke, sings, conversion ")
print("===== All flows ===")
print(numbers)
print("===== Bi-direcitonal only ===")
print(numbers2)