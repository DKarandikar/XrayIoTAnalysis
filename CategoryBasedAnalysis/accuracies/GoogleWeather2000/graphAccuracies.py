import pickle, os
import numpy as np
import matplotlib.pyplot as plt

test = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'testAccuracies.p'),'rb'))
train = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'trainAccuracies.p'),'rb'))

x = np.arange(len(test)) 

plt.plot(x, test)
plt.plot(x, train)

plt.show()