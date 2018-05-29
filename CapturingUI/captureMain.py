import os, csv, _thread, threading, asyncio, tkinter, time, pickle
import numpy as np
import pandas as pd
import pyshark
from tkinter import Tk, RIGHT, BOTH, RAISED, LEFT
from tkinter.ttk import Frame, Button, Style, Scrollbar
from collections import defaultdict
from scapy.all import *

PCAPS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pcaps")

INTERFACE_NAME = r"\Device\NPF_{5DB2AC8D-8B1C-46CD-8217-721669155FF0}"
INTERFACE_NAME = '2'

HOME_NET_PREFIX = "192.168.1"

"""
for packet in capture.sniff_continuously():
    print(packet)
"""


capture = pyshark.LiveCapture(interface=INTERFACE_NAME, only_summaries=False)

class InfoBar(Frame):
    def __init__(self, parent, keyLabel, *args, **kwargs):
        Frame.__init__(self, parent)
        self.label = tkinter.Label(self, text=keyLabel)
        self.text1 = tkinter.Entry(self)
        self.text2 = tkinter.Entry(self)
        self.button1 = Button(self, text="Save", command=lambda: parent.saveCaptures(keyLabel, self.text1.get(), self.text2.get()))

        self.label.pack(side=LEFT)
        self.text1.pack(side=LEFT)
        self.text2.pack(side=LEFT)
        self.button1.pack(side=LEFT)

        self.name = keyLabel

class Capturing(Frame):
    
    def saveCaptures(self, IP, device, action):
        self.saving = True
        counter = 0
        while True:
            if os.path.isfile(os.path.join(PCAPS_PATH, device + action + counter + ".p")):
                counter += 1
            else:
                break
        pickle.dump(self.IPDict[IP], open(os.path.join(PCAPS_PATH, device + action + counter + ".p"), "wb") )
        self.saving = False
    
    def start(self, click):
        if click:
            self.wantToSniff = True

        if self.running:
            self.tick()
            self.after(50,lambda: self.start(False))
    
    def stop(self):
        self.wantToSniff = False
  
    def __init__(self):
        super().__init__()   

        self.running = True
        self.wantToSniff = False
        self.Sniffing = False

        self.IPDict = defaultdict(list)
        self.infobars = []

        self.first = True
        self.saving = False

        self.packetCount = tkinter.IntVar()
        self.packetCount.set(0)

        # Do this last
        self.updateUI()

    def updateUI(self):
      
        self.master.title("Buttons")
        self.style = Style()
        self.style.theme_use("default")

        if self.first:
            tkinter.Label(self, textvariable=self.packetCount).pack()

        for key in self.IPDict.keys():
            ignore = False
            for frame in self.infobars:
                if frame.name == key:
                    ignore = True
            if not ignore:
                frame = InfoBar(self, key, relief=RAISED, borderwidth=1)
                self.infobars.append(frame)
                frame.pack(side="top", fill=BOTH, expand=True)
        
        self.pack(fill=BOTH, expand=True)

        if self.first:
        
            stopButton = Button(self, text="Stop Scan", command=self.stop)
            stopButton.pack(side=RIGHT, padx=5, pady=5)
            startButton = Button(self, text="Start Scan", command=lambda: self.start(True))
            startButton.pack(side=RIGHT)
    
        # Do this last
        self.first = False

    def packetSniff(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        capture.setup_eventloop()

        for packet in capture.sniff_continuously(packet_count=100):
            self.processPacket(packet)
            self.packetCount.set(self.packetCount.get() + 1)
        
        self.Sniffing = False

    def tick(self):
        if not self.Sniffing and self.wantToSniff:
            threading.Thread.__init__(self)
            t = threading.Thread( target=self.packetSniff, name="Sniffing")
            t.start()
            self.Sniffing = True
            

        #print(self.IPDict)
        self.updateUI()
        
    def processPacket(self, packet):
        while self.saving:
            time.sleep(1)
        try:
            if HOME_NET_PREFIX in packet.ip.src:
                self.IPDict[packet.ip.src].append(packet)
        except AttributeError:
            # This can happen if a packet has no IP layer
            # We will just ignore packets like this
            print("Attribute Error")
            

def main():
  
    root = Tk()
    root.geometry("800x600+300+300")
    app = Capturing()
    root.mainloop()  


if __name__ == '__main__':
    main()  
