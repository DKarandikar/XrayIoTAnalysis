import threading, asyncio, tkinter, time, json, os
import pyshark
from tkinter.ttk import Frame, Button, Style, Scrollbar
from collections import defaultdict

import infoBar, packetProcessing, inforBarHeadings

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"config.json")) as f:
    data = json.load(f)

    INTERFACE_NAME = data["interface_name"]
    HOME_NET_PREFIX = data["home_ip"]
    summaries = data["only_summaries"] == "True"
    
PACKET_LABEL_TEXT = "Total packets so far: "

capture = pyshark.LiveCapture(interface=INTERFACE_NAME, only_summaries=summaries)

class Capturing(Frame):
    
    def saveCaptureButtonFUN(self, IP, device, action):
        """
        Saves the packets stored since start or last save for the ip IP
        and records device name and action from Entry Boxes
        """
        self.saving = True
        self.IPDict = packetProcessing.savingPackets(IP, device, action, self.IPDict)
        self.saving = False

    def resetButtonFUN(self, IP, device, action):
        """
        Clears the IP packets for that row
        """
        self.saving = True
        self.IPDict.pop(IP, None)
        infobar = self.getInfobar(IP)
        infobar.updatePacketCount(0)
        self.saving = False
    
    def startButtonFUN(self, click):
        """
        Starts scanning and updating the UI
        """
        if click:
            self.wantToSniff = True

        self.tick()
        self.after(50,lambda: self.startButtonFUN(False))
    
    def stopButtonFUN(self):
        """
        Tells pyshark to stop scanning
        """
        self.wantToSniff = False
  
    def __init__(self):
        """
        Setup all the class variables 
        """
        super().__init__()   

        self.wantToSniff = False
        self.Sniffing = False

        self.IPDict = defaultdict(list)
        self.infobars = []

        self.saving = False

        self.packetCountStringVar = tkinter.StringVar()
        self.packetCountStringVar.set(PACKET_LABEL_TEXT)
        self.packetCount = 0

        # Do this last
        self.initUI()

    def initUI(self):
        """
        Sets up the initial UI componenents
        These are the ones that won't change during use
        """
        
        self.master.title("Packet Capture")
        self.style = Style()
        self.style.theme_use("default")

        tkinter.Label(self, textvariable=self.packetCountStringVar).grid(row=0, column=0, columnspan=2)

        separator = Frame(self, height=2, relief=tkinter.SUNKEN)
        separator.grid(row=1, column=0, columnspan=2)  

        self.infoBarFrame = tkinter.Frame(self)
        self.infoBarFrame.grid(row=2, column=0, columnspan=2, padx=5, pady=20)

        headings = inforBarHeadings.InfoBarHeadings(self.infoBarFrame)
        headings.pack(side="top", fill=tkinter.BOTH, expand=False)
 
        stopButton = Button(self, text="Stop Scan", command=self.stopButtonFUN)
        stopButton.grid(row=3, column=1)
        startButton = Button(self, text="Start Scan", command=lambda: self.startButtonFUN(True))
        startButton.grid(row=3, column=0)

        self.pack()
    
        self.updateUI()

    def updateUI(self):
        """
        Updates the UI components that change as it goes along
        """
        
        # Loop through all IP addresses
        for key in self.IPDict.keys():
            ignore = False

            # Check if we already have an infobar for that IP 
            for infobar in self.infobars:
                if infobar.name == key:
                    infobar.updatePacketCount(len(self.IPDict[key]))
                    ignore = True

            # If we don't, make a new one and add it
            if not ignore:
                infobar = infoBar.InfoBar(self.infoBarFrame, self, key, relief=tkinter.RAISED, borderwidth=1)
                self.infobars.append(infobar)
                infobar.pack(side="top", fill=tkinter.BOTH, expand=False)
      
        
    def packetSniff(self):
        """
        Sniffing method that runs on another thread
        Runs forever until the stop button is pressed
        """

        # Have to do this to keep pyshark happy
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        capture.setup_eventloop()

        for packet in capture.sniff_continuously():
            # Don't want to mess when saving to avoid race conditions
            while self.saving:
                time.sleep(1)
            self.IPDict = packetProcessing.processPacket(packet, self.IPDict)
            self.packetCount += 1
            self.packetCountStringVar.set(PACKET_LABEL_TEXT + str(self.packetCount))

            if not self.wantToSniff:
                break
        
        self.Sniffing = False

    def tick(self):
        """
        Tick that runs every 50ms
        Calls the sniffing thread if necessary, and otherwise just updates UI
        """
        if not self.Sniffing and self.wantToSniff:
            threading.Thread.__init__(self)
            t = threading.Thread( target=self.packetSniff, name="Sniffing")
            t.start()
            self.Sniffing = True

        self.updateUI()

    def getInfobar(self, IP):
        """ Gets infobar for IP """
        for infobar in self.infobars:
            if infobar.name == IP:
                return infobar
        

            

def main():
  
    root = tkinter.Tk()
    root.geometry("580x400+300+300")
    app = Capturing()
    root.mainloop()  


if __name__ == '__main__':
    main()  
