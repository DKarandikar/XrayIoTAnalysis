# Capturing UI

A simple python program to assist in capturing and pickling packets from different local IPs on a network.

To use, run `captureMain.py`

Config is stored in the `config.json` file, this should be customised to the network and interface desired.

## Pyshark notes

The latest versions of pyshark don't work well on windows (they won't capture packets) it is better ot roll-back to version 0.3.6.2

A further issue is with pickling, that issue and the solution to it if the `save` button isn't working can be found here: https://github.com/KimiNewt/pyshark/issues/63