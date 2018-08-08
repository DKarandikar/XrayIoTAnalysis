"""
Not used anymore, min/max normalised a set of data 
Now we do the normalisation as part of the PCA step 
"""
import os
import pandas as pd

NUMBER_COLUMNS = 56
NUMBER_FEATURES = 54

ORIGINAL_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "FlowFeatures.csv")
OUTPUT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "normalizedEleven.csv")

df = pd.read_csv( ORIGINAL_DATA, usecols = [x for x in range(NUMBER_COLUMNS-NUMBER_FEATURES,NUMBER_COLUMNS)], header=None)
df2 = pd.read_csv( ORIGINAL_DATA, usecols = [x for x in range(0,NUMBER_COLUMNS-NUMBER_FEATURES)], header=None)

normalized_df = (df-df.min())/(df.max()-df.min()+0.000000000000000001)

#print(normalized_df)

big = pd.merge(df2, normalized_df, on=df2.index, how='inner')

del big['key_0']

print(big)

big.to_csv( OUTPUT_DATA, index = False, header=None) 