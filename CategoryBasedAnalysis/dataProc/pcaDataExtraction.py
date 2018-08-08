"""
Script to perform PCA on data 
"""
import os, pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ORIGINAL_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "GoogleFlowfeaturesBig.csv")
OUTPUT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "normalizedPCAGoogleWeatherOut.csv")

# Read in the data 
unNomralized = np.genfromtxt(ORIGINAL_DATA, delimiter=",", usecols=[x for x in range(2,56)] ) #+[x for x in range(57, 56 + 18)]
scaler = StandardScaler()
scaler.fit(unNomralized)
normalized = scaler.transform(unNomralized)

# Perform PCA on it
pca = PCA(n_components=18)

# 12 components gets 0.97890031 of the variance using .cumsum()
# 12 for weather Alexa gives 0.97050849 of variance 
# 6 components for weather Alexa OUT only gives 0.97186234
# 18 components gets 0.97605086 for google home of the variance using .cumsum()
# 24 for delta Data gives 0.97640022
# 14 for delta Out gives 0.97685787
# 16 for Google weather all gives 0.97162899
# 8 for google weather out gives 0.97519131

#pca = PCA()#

principalComponents = pca.fit_transform(normalized)

print(pca.explained_variance_ratio_.cumsum())

# Rescale the data after transformation
scaler2 = StandardScaler()
scaler2.fit(principalComponents)
principalDf = pd.DataFrame(data = scaler2.transform(principalComponents))

# Add the titles and classes back on and output ot a csv
df2 = pd.read_csv( ORIGINAL_DATA, usecols = [x for x in range(0,2)], header=None)
big = pd.merge(df2, principalDf, on=df2.index, how='inner')
del big['key_0']
print(big)

### Do one of the following, or both:

# Output data to csv
#big.to_csv( OUTPUT_DATA, index = False, header=None) 

# Output scalers for use in predicting with models elsewhere 
pickle.dump( pca, open( "pca.p", "wb" ) )
pickle.dump( scaler, open( "preScaler.p", "wb" ) )
pickle.dump( scaler2, open( "postScaler.p", "wb" ) )