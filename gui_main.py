from tkinter import *
from PIL import Image, ImageDraw

import numpy as np

from keras.models import load_model

import time
import math
from scipy import ndimage
import cv2

from convert_mnist_format import ConvertMNISTFormat

class PaintWindow():
    
    predict_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

    def __init__(self, master, model):
        self.last_x, self.last_y = None, None
        self.image_number = 0
        self.model = model

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
        self.image = self.image.resize((28, 28), Image.ANTIALIAS)
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
#        self.text.insert(INSERT, '-something')

    # Function to execute when the motion event has ended.
    def motion_end(self, event):
#        self.predict_char(self.image)
        self.text.insert(INSERT, '-something')
        self.clear_canvas()

    def clear_canvas(self):
        time.sleep(0.5)
        self.cv.delete('all')
        self.save()
        print(self.image.size)
        for x in range(self.image.size[0]):
            for y in range(self.image.size[1]):
                self.image.putpixel((x, y), (255))
#        self.save()
        
    def predict_char(self, image):
        image = image.resize((28, 28))
        image_arr = np.asarray(image)
        preprocess = ConvertMNISTFormat(image)
        image_arr = preprocess.process_image()
        image_arr = np.divide(image_arr, 255)
        image_arr = np.resize(image_arr, (1, 28, 28, 1))
        predict = self.model.predict(image_arr)
        print(np.argmax(predict[0]))
#        return predict_classes[predict]

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

# Load the saved pre-trained model
model = load_model('cnn-digits.h5')


cv = PaintWindow(root, model)

root.mainloop()
