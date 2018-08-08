"""
Uses various predictors to evaluate feature importance
Sample output is in a comment below
"""
import os, pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier

NUMBER_COLUMNS = 56
ORIGINAL_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataProc", "data",  "FlowFeatures.csv")

NUMBER_COLUMNS = 60
ORIGINAL_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataProc", "data",  "FlowFeaturesTime.csv")

def fitAndPrint(classifier, data, target):
    """ 
    Fit classifier to target data and print the best and worst features
    """
    classifier.fit(data, target)
    importances = classifier.feature_importances_

    indices = np.argsort(importances)[::-1]

    featureNames = []

    if NUMBER_COLUMNS == 60:
        featureNames.append("QuestionTime")
        featureNames.append("ResponseTime")
        featureNames.append("OutgoingMaxGapTime")
        featureNames.append("IncomingMaxGapTime")

    for z in ["OUT", "IN", "BOTH"]:
        
        featureNames.append(z + "Min")
        featureNames.append(z + "Max")
        featureNames.append(z + "Mean")
        featureNames.append(z + "Mad")
        featureNames.append(z + "Std")
        featureNames.append(z + "Var")
        featureNames.append(z + "Skew")
        featureNames.append(z + "Kurtosis")

        for value in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            featureNames.append(z + "Quantile" + str(value))
        featureNames.append(z + "Number")

    results = []
    for x in indices:
        results.append(featureNames[x])

    outImportance = sum(importances[:18])
    inImportance= sum(importances[18:36])
    bothImportance = sum(importances[36:54])

    print("For "  + str(type(classifier).__name__) + "the OUT/IN/BOTH importances summed are %.2f%% / %.2f%% / %.2f%%" % (outImportance*100, inImportance*100, bothImportance*100))

    print("Top 10 most important features in order for " + str(type(classifier).__name__) + " are as follows:")
    print("")

    for x in range(10):
        print(results[x] + " with percentage %.2f%%" % (float(importances[indices[x]]) *100) )
    print("")

    print("Bottom 5 most unimportant features in order for " + str(type(classifier).__name__) + " are as follows:")
    print("")

    for x in range(NUMBER_COLUMNS - 2 - 5, NUMBER_COLUMNS - 2):
        print(results[x] + " with percentage %.2f%%" % (float(importances[indices[x]]) *100) )
    print("")



# Read in the data 
unNomralized = np.genfromtxt(ORIGINAL_DATA, delimiter=",", usecols=[x for x in range(2,NUMBER_COLUMNS)])
scaler = StandardScaler()
scaler.fit(unNomralized)
normalized = scaler.transform(unNomralized)

y = np.genfromtxt(ORIGINAL_DATA, delimiter=",", usecols=1)

forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
forest2 = RandomForestClassifier(n_estimators=250, random_state=0)
gradient = GradientBoostingClassifier(n_estimators=100, random_state=0)
#ada = AdaBoostClassifier(n_estimators=1000, random_state=0)    # ada gave NaN several times and so was ignored 

fitAndPrint(forest, normalized, y)
fitAndPrint(forest2, normalized, y)
fitAndPrint(gradient, normalized, y)
#fitAndPrint(ada, normalized, y) 

"""
For ExtraTreesClassifierthe OUT/IN/BOTH importances summed are 32.12% / 39.53% / 28.35%
Top 10 most important features in order for ExtraTreesClassifier are as follows:

BOTHNumber with percentage 6.00%
INNumber with percentage 5.55%
OUTNumber with percentage 5.14%
INStd with percentage 4.47%
INVar with percentage 3.99%
INMad with percentage 3.94%
INMean with percentage 3.84%
INSkew with percentage 3.76%
INKurtosis with percentage 3.52%
OUTVar with percentage 3.51%

Bottom 5 most unimportant features in order for ExtraTreesClassifier are as follows:

BOTHQuantile0.2 with percentage 0.02%
OUTMin with percentage 0.02%
INQuantile0.1 with percentage 0.01%
INQuantile0.2 with percentage 0.01%
INMin with percentage 0.01%

For RandomForestClassifierthe OUT/IN/BOTH importances summed are 29.55% / 43.29% / 27.17%
Top 10 most important features in order for RandomForestClassifier are as follows:

BOTHNumber with percentage 6.74%
INSkew with percentage 5.63%
INNumber with percentage 5.38%
INKurtosis with percentage 5.03%
INMad with percentage 5.03%
INMean with percentage 4.87%
OUTNumber with percentage 4.65%
INVar with percentage 4.49%
INStd with percentage 4.47%
OUTMad with percentage 3.89%

Bottom 5 most unimportant features in order for RandomForestClassifier are as follows:

INQuantile0.3 with percentage 0.01%
INQuantile0.2 with percentage 0.01%
INMin with percentage 0.00%
BOTHMin with percentage 0.00%
OUTMin with percentage 0.00%

For GradientBoostingClassifierthe OUT/IN/BOTH importances summed are 34.21% / 38.12% / 27.67%
Top 10 most important features in order for GradientBoostingClassifier are as follows:

INNumber with percentage 6.28%
INKurtosis with percentage 5.88%
BOTHNumber with percentage 5.65%
INSkew with percentage 5.18%
OUTMean with percentage 4.64%
OUTNumber with percentage 4.64%
BOTHSkew with percentage 4.40%
OUTMad with percentage 4.17%
INMean with percentage 4.08%
BOTHMad with percentage 3.91%

Bottom 5 most unimportant features in order for GradientBoostingClassifier are as follows:

INQuantile0.3 with percentage 0.00%
BOTHMin with percentage 0.00%
BOTHQuantile0.1 with percentage 0.00%
BOTHQuantile0.2 with percentage 0.00%
OUTMin with percentage 0.00%

For AdaBoostClassifierthe OUT/IN/BOTH importances summed are nan% / 25.20% / 31.20%
Top 10 most important features in order for AdaBoostClassifier are as follows:

OUTSkew with percentage nan%
OUTVar with percentage nan%
BOTHNumber with percentage 7.20%
OUTMean with percentage 6.40%
OUTKurtosis with percentage 4.80%
INQuantile0.7 with percentage 4.40%
BOTHKurtosis with percentage 4.40%
OUTNumber with percentage 4.00%
INStd with percentage 4.00%
OUTQuantile0.4 with percentage 3.60%

Bottom 5 most unimportant features in order for AdaBoostClassifier are as follows:

BOTHMax with percentage 0.00%
BOTHQuantile0.1 with percentage 0.00%
OUTMin with percentage 0.00%
INQuantile0.4 with percentage -0.00%
BOTHQuantile0.2 with percentage -0.00%
"""