from tkinter import *
from PIL import Image, ImageDraw

import time

class PaintWindow():

    def __init__(self, master):
        self.last_x, self.last_y = None, None
        self.image_number = 0

        # Canvas to draw upon
        self.cv = Canvas(master, width=500, height=500, bg='white')
        self.cv.bind('<1>', self.activate_paint)
        self.cv.grid(row=0, column=0, sticky=W)

        # A frame inside the master for scroll bar.
        self.frame_inner = Frame(master, width=400, height=500)
        self.frame_inner.grid(row=0, column=2, sticky=E)

        # Text to display identified text.
        self.text = Text(self.frame_inner, bg='white', fg='black', bd=10)
        self.text.pack(side=LEFT)

        # Scrollbar for Text
        self.scrollbar = Scrollbar(self.frame_inner, command = self.text.yview)
        self.scrollbar.pack(side=RIGHT, fill=Y)
        self.text['yscrollcommand'] = self.scrollbar.set

        # --PIL---
        self.image = Image.new('L', (500, 500), 'white')
        self.draw = ImageDraw.Draw(self.image)

        # A frame to hold the buttons
        self.frame_btns = Frame(master)
        self.frame_btns.grid(row=1)

        # Buttons for some operations.
        self.btn_space = Button(self.frame_btns, text='space', command=self.insert_space)
        self.btn_space.grid(row=0, column=0)

        self.btn_full_stop = Button(self.frame_btns, text='.', command=self.insert_fullstop)
        self.btn_full_stop.grid(row=0, column=1)

        self.btn_question_mark = Button(self.frame_btns, text='?', command=self.insert_question_mark)
        self.btn_question_mark.grid(row=0, column=2)

        self.btn_comma = Button(self.frame_btns, text=',', command=self.insert_comma)
        self.btn_comma.grid(row=0, column=3)

        self.btn_new_line = Button(self.frame_btns, text='new_line', command=self.insert_new_line)
        self.btn_new_line.grid(row=0, column=4)

        self.btn_exclamation = Button(self.frame_btns, text='!', command=self.insert_exclamation)
        self.btn_exclamation.grid(row=0, column=5)

    def save(self):
        filename = 'image_{}.png'.format(self.image_number)
        self.image.save(filename)
        self.image_number += 1

    def activate_paint(self, event):
        self.cv.bind('<B1-Motion>', self.paint)
        self.cv.bind('<ButtonRelease-1>', self.motion_end)
        self.last_x, self.last_y = event.x, event.y

    def paint(self, event):
        x, y = event.x, event.y
        self.cv.create_line((self.last_x, self.last_y, x, y), width=4)
        # --PIL--
        self.draw.line((self.last_x, self.last_y, x, y), fill='black', width=4)
        self.last_x, self.last_y = x, y
        # Update label text
        self.text.insert(INSERT, '-something')

    # Function to execute when the motion event has ended.
    def motion_end(self, event):
        self.clear_canvas()

    def clear_canvas(self):
        time.sleep(0.5)
        self.cv.delete('all')
        self.save()
        for x in range(500):
            for y in range(500):
                self.image.putpixel((x, y), (255))
        self.save()

    def insert_space(self):
        self.text.insert(INSERT, ' ')

    def insert_fullstop(self):
        self.text.insert(INSERT, '. ')

    def insert_question_mark(self):
        self.text.insert(INSERT, '? ')

    def insert_comma(self):
        self.text.insert(INSERT, ', ')

    def insert_exclamation(self):
        self.text.insert(INSERT, '! ')

    def insert_new_line(self):
        self.text.insert(INSERT, '\n')


root = Tk()

cv = PaintWindow(root)

root.mainloop()
