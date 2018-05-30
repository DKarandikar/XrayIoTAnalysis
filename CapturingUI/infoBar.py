import tkinter
from tkinter import Tk, RIGHT, BOTH, RAISED, LEFT
from tkinter.ttk import Frame, Button, Style, Scrollbar

class InfoBar(Frame):
    def __init__(self, parent, parentParent, keyLabel, *args, **kwargs):
        Frame.__init__(self, parent)
        self.label = tkinter.Label(self, text=keyLabel)
        self.text1 = tkinter.Entry(self)
        self.text2 = tkinter.Entry(self)
        self.button1 = Button(self, text="Save", command=lambda: parentParent.saveCaptureButtonFUN(keyLabel, self.text1.get(), self.text2.get()))
        self.button2 = Button(self, text="Reset", command=lambda: parentParent.resetButtonFUN(keyLabel, self.text1.get(), self.text2.get()))
        self.label2 = tkinter.Label(self, text=" ")

        self.label.pack(side=LEFT)
        self.text1.pack(side=LEFT)
        self.text2.pack(side=LEFT)
        self.button1.pack(side=LEFT)
        self.button2.pack(side=LEFT)
        self.label2.pack(side=LEFT)

        self.name = keyLabel

    def updatePacketCount(self, val):
        self.label2.config(text="Packets: " + str(val))