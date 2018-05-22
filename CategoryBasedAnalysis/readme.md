## Pipeline

1. Get `.pcap` bursts and place in bursts/
2. Run `BurstFlowFeatureExtraction.py` to get all flow data in `flowFeatures.csv` (this is the one that takes time)
3. (NO Longer Required) Run `RemoveNonBiDirectionalFlows.py` to remove any flows that don't form a conversation, and thus remove any `nan` issues
4. Run `normaliseData.py` to min/max normalize all columns into [0,1]
5. Run the `basicNN.py` to train on the extracted flows

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
