# Automated capture

Place m4a files with recordings of any length of questions with pauses for answers in /audioFiles

`automateCap.py` plays every file in /audioFiles, loops each up to 20 mins, captures the traffic and saves to a .pcap and also performs statistical analysis on the lengths and stores the results in a csv file in /data

## Time

To use the extra statistics from `automatedTime.py` place .wav files in /audioCutUps each of which has only the question and no wait time (`splitAudioFiles.py` does a good job of automating this given input .wav files of the above form)

Running `automatedTime.py` does similar but also records the response, and gets the time from that, records the duration of the question, and also the time between first and last packet of maximum size outgoing and incoming 

Hence the csv contains name/class/questionTime/responseTime/outgoingMaxPacketTime/incomingMaxPacketTime/normal statistics

Look at `maxLengthPacketTimes` in `statisticsProcessing.py` to understand more what is meant by outgoingMaxPacketTime and incomingMaxPacketTime