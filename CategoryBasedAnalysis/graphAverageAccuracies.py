import pickle, os, math
import numpy as np
import matplotlib.pyplot as plt

BIN_NUMBER = 25

## Get data in 

test = np.array(pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'testAccuracies.p'),'rb')))
train = np.array(pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'trainAccuracies.p'),'rb')))

def strided_app(a, L, S ):
    """ Get a strided array from array a with length L and stride S """
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

ticks = np.arange(len(test) ) 

strided = strided_app(test, L=BIN_NUMBER, S=1)
strided_train = strided_app(train, L=BIN_NUMBER, S=1)

## Average into BIN_NUMBER where we can, but this will always be less values 

y = [np.mean(strided[x]) for x in range(len(test) - BIN_NUMBER + 1)]
z = [np.mean(strided_train[x]) for x in range(len(test) - BIN_NUMBER + 1)]

## Half before, add average of data to that point in 5 element block 

for k in range(1, math.floor((BIN_NUMBER-1)/2)):
    y = np.insert(y, 0+k-1, np.mean(test[0+k-5 if 0+k-5>0 else 0:k]), axis=0)
    #print(test[0:k])
    z = np.insert(z, 0+k-1, np.mean(train[0+k-5 if 0+k-5>0 else 0:k]), axis=0)

## Add one more entry to account for 0:0 not being a slice

y = np.insert(y, 0, y[0], axis=0)
z = np.insert(z, 0, z[0], axis=0)

## Make list so we can append 

y = list(y)
z = list(z) 

## Half after, just add the last value to give a short straight line at end 

for _ in range(math.ceil((BIN_NUMBER-1)/2)):
    y.append(y[-1])
    z.append(z[-1])

## Plot and show 

plt.plot(ticks, test)
plt.plot(ticks, train)
plt.plot(ticks, y)
plt.plot(ticks, z)

plt.show()