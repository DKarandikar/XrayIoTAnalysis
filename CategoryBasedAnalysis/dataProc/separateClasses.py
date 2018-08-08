"""
Gets only the classes specified in CLASSES
"""
import os, csv

CLASSES = [5,6]

output_filename = "normalizedClasses"
for x in CLASSES:
    output_filename += str(x)
output_filename += ".csv"

NORMALIZED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "normalized.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  output_filename)


with open(NORMALIZED_FILE,"r") as feat:
    with open(OUTPUT_FILE, "a", newline='') as output:
        features = csv.reader(feat)
        result = csv.writer(output)
        for row in features:
            if row:
                if int(row[1]) in CLASSES:
                    result.writerow(row)
            
print("Done")