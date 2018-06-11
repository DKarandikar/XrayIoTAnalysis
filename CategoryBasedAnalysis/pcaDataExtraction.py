import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ORIGINAL_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "FlowFeatures.csv")
OUTPUT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "normalizedPCA.csv")

# Read in the data 
unNomralized = np.genfromtxt(ORIGINAL_DATA, delimiter=",", usecols=[x for x in range(2,56)])
scaler = StandardScaler()
scaler.fit(unNomralized)
normalized = scaler.transform(unNomralized)

# Perform PCA on it
pca = PCA(n_components=12)
# 12 components gets 0.97890031 of the variance using .cumsum()
principalComponents = pca.fit_transform(normalized)

# Rescale the data after transformation
scaler2 = StandardScaler()
scaler2.fit(principalComponents)
principalDf = pd.DataFrame(data = scaler2.transform(principalComponents))

# Add the titles and classes back on and output ot a csv
df2 = pd.read_csv( ORIGINAL_DATA, usecols = [x for x in range(0,2)], header=None)
big = pd.merge(df2, principalDf, on=df2.index, how='inner')
del big['key_0']
print(big)
big.to_csv( OUTPUT_DATA, index = False, header=None) 