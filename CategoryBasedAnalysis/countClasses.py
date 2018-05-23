import os, csv

numbers = [0 for x in range(10)]
numbers2 = [0 for x in range(10)]

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "normalized.csv"),"r") as feat:
    features = csv.reader(feat)
    for row in features:
        if row:
            numbers[int(row[1])] += 1

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "FlowFeatures.csv"),"r") as feat:
    features = csv.reader(feat)
    for row in features:
        if row:
            numbers2[int(row[1])] += 1

print()
print( "Noise, time, weather, joke, sings, conversion, day, timer, shopping ")
print("===== All normalized ===")
print(numbers)
print("===== All flows ===")
print(numbers2)