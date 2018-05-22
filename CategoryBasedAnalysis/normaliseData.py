import os
import pandas as pd

NUMBER_COLUMNS = 56

df = pd.read_csv( os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "noNonBiDirect.csv"), usecols = [x for x in range(4,NUMBER_COLUMNS)], header=None)
df2 = pd.read_csv( os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "noNonBiDirect.csv"), usecols = [x for x in range(0,4)], header=None)
normalized_df = (df-df.min())/(df.max()-df.min()+0.000000000000000001)

print(normalized_df)

big = pd.merge(df2, normalized_df, on=df2.index, how='inner')

del big['key_0']

print(big)

big.to_csv( os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",  "normalized.csv"), index = False, header=None) 