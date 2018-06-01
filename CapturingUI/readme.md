# Capturing UI

A simple python program to assist in capturing and pickling packets from different local IPs on a network.

Designed for python 3

To use, run `captureMain.py`

Config is stored in the `config.json` file, this should be customised to the network and interface desired.

## Pyshark notes

The latest versions of pyshark don't work well on windows (they won't capture packets) it is better to roll-back to version 0.3.6.2

To fix this run `pip install pyshark==0.3.6.2`

A further issue is with pickling, that issue and the solution to it if the `save` button isn't working can be found here: https://github.com/KimiNewt/pyshark/issues/63

A final thing to note is that TShark has to be functioning for users (non-admin) on this device, information about this can be found at https://wiki.wireshark.org/CaptureSetup/CapturePrivileges

## Processing Captures

Use `python procGetAllIPs.py` to output a json file containing all the IPs that are source or destination for each capture
Use `python procPacketBurstification.py` to output pickled packet files where all packets are within 1 sec of the most recent one

Use `node procWhoIs.js` to output a json file containing all the company names for the IPs outputted above