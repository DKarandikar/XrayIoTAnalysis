import tkinter
from tkinter import Tk, RIGHT, BOTH, RAISED, LEFT
from tkinter.ttk import Frame, Button, Style, Scrollbar

class InfoBarHeadings(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent)
        self.label = tkinter.Label(self, text="IP Address")
        self.label2 = tkinter.Label(self, text="Device Name")
        self.label3 = tkinter.Label(self, text="Action Name")
        self.label4 = tkinter.Label(self, text="Save/Reset?")
        self.label5 = tkinter.Label(self, text="Packets So Far")
        

        self.label.pack(side=LEFT)
        separator = Frame(self, height=2, relief=tkinter.SUNKEN)
        separator.pack(side=LEFT, padx=15, pady=5)
        self.label2.pack(side=LEFT)
        separator = Frame(self, height=2, relief=tkinter.SUNKEN)
        separator.pack(side=LEFT, padx=25, pady=5)
        self.label3.pack(side=LEFT)
        separator = Frame(self, height=2, relief=tkinter.SUNKEN)
        separator.pack(side=LEFT, padx=25, pady=5)
        self.label4.pack(side=LEFT)
        separator = Frame(self, height=2, relief=tkinter.SUNKEN)
        separator.pack(side=LEFT, padx=25, pady=5)
        self.label5.pack(side=LEFT)
        