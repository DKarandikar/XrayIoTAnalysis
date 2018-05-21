## Pipeline

1. Get pcap bursts and place in bursts/
2. Run `BurstFlowFeatureExtraction.py` to get all flow data in flowFeatures.csv
3. Run `RemoveNonBiDirectionalFlows.py` to remove any flows that don't form a conversation, and thus remove any `nan` issues
4. Run the basicNN.py to train on the extracted flows