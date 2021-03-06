"""
Counts the classes in a csv file
Gives values of how many of each category is used in a given set of statistics 
"""
import os, csv

numbers = [0 for x in range(15)]
numbers2 = [0 for x in range(15)]

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "GoogleFlowfeatures.csv"),"r") as feat:
    features = csv.reader(feat)
    for row in features:
        if row:
            numbers[int(row[1])] += 1

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "GoogleFlowfeatures.csv"),"r") as feat:
    features = csv.reader(feat)
    for row in features:
        if row:
            numbers2[int(row[1])] += 1

print()
print( "Noise, time, weather, joke, sings, conversion, day, timer, shopping, lightsOnOff, lightsBrightDim, alarms ")
print("===== All normalized ===")
print(numbers)
print("===== All flows ===")
print(numbers2)