import tkinter
from tkinter import Tk, RIGHT, BOTH, RAISED, LEFT
from tkinter.ttk import Frame, Button, Style, Scrollbar

class InfoBar():
    def __init__(self, parent, parentParent, keyLabel, *args, **kwargs):
        self.label = tkinter.Label(parent, text=keyLabel)
        self.text1 = tkinter.Entry(parent)
        self.text2 = tkinter.Entry(parent)
        self.button1 = Button(parent, text="Save", command=lambda: parentParent.saveCaptureButtonFUN(keyLabel, self.text1.get(), self.text2.get()))
        self.button2 = Button(parent, text="Reset", command=lambda: parentParent.resetButtonFUN(keyLabel, self.text1.get(), self.text2.get()))
        self.label2 = tkinter.Label(parent, text=" ")

        self.name = keyLabel

    def updatePacketCount(self, val):
        self.label2.config(text="Packets: " + str(val))