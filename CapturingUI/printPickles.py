import os, pickle


PCAPS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pcaps")

f = []
for (dirpath, dirnames, filenames) in os.walk(PCAPS_PATH):
    f.extend(filenames)
    break

for file in f:
    k = pickle.load(open(os.path.join(PCAPS_PATH, file), "rb"))
    print(len(k))

