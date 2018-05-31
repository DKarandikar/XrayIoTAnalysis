import tkinter
from tkinter import Tk, RIGHT, BOTH, RAISED, LEFT
from tkinter.ttk import Frame, Button, Style, Scrollbar

class InfoBarHeadings():
    def __init__(self, parent, *args, **kwargs):
        self.label = tkinter.Label(parent, text="IP Address")
        self.label2 = tkinter.Label(parent, text="Device Name")
        self.label3 = tkinter.Label(parent, text="Action Name")
        self.label4 = tkinter.Label(parent, text="Save/Reset?")
        self.label5 = tkinter.Label(parent, text="Packets So Far")

        