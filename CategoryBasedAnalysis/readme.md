## Pipeline

1. Get `.pcap` bursts and place in bursts/ (or else get `pcaps` and run `PcapBurstification.py`)
2. Run `BurstFlowFeatureExtraction.py` to get all flow data in `flowFeatures.csv` (this is the one that takes time)
3. Run `normaliseData.py` to min/max normalize all columns into [0,1]
4. Run the `basicNN.py` to train on the extracted flows

## Categories

Category Numbers are:
1. Time
2. Weather 
3. Joke
4. Song author
5. Conversion 
6. Day Of Week
7. Timer
8. Shopping 

## Other scripts

`allClassF1Scores.py` takes the `normalised.csv` file and runs each class against all the others as a binary classification problem and then calculates the F1-score and saves that to a file (and prints to console) 

`testPcap.py` can be used to run a trained model against a new Pcap file to see how it performs

`randomiseClasses.py` produces a data csv with all class values randomized

`getOnlyAlexaData.oy` produces a data csv with only incoming traffic from `normalised.csv`

`countClasses.py` counts the number of each data flow in the normalized and non-normalized data files 

`saveMinMax.py` saves the min and max of each column in `FlowFeatures.csv` for use with the live evaluation script in xray (i.e. the one that runs the trained model live against a running Echo device)

`loadNNTest.py` loads a trained model and displays the confusion matrix and accuracy and also pickles the values of the weights, to be used live via `numpy.matmul` so that it is far quicker than Tensorflow  

`separateClasses.py` can be used to extract only certain classes from `normalised.csv` to train on a subset of that data

