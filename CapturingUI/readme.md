# Capturing UI

A simple python program to assist in capturing and pickling packets from different local IPs on a network.

Designed for python 3

To use, run `captureMain.py`

Config is stored in the `config.json` file, this should be customised to the network and interface desired.

Uses Scapy to capture the packets, correct interface name for the config file can be found by running `ifaces` in a python interpreter 

## Processing Captures

Use `python procGetAllIPs.py` to output a json file containing all the IPs that are source or destination for each capture

Use `python procPacketBurstification.py` to output pickled packet files where all packets are within 1 sec of the most recent one

Use `node procWhoIs.js` to output a json file containing all the company names for the IPs outputted above