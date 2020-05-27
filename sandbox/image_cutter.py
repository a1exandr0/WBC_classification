import PIL.Image
from PIL import ImageTk
from tkinter import *
import cv2 as cv
import numpy as np
import os

path_base = "/Users/a1exandr0/PycharmProjects/CourseWork/ImageAugment/LISC Database/Main Dataset/lymp/"
save_base = "/Users/a1exandr0/PycharmProjects/CourseWork/SelfCutData/Lymphocyte/lymphocyte_{}_augment_n_{}.png"
counter_global = 0
counter_help = 0
cuts_per_im = 5
pics = os.listdir(path_base)
for i, el in enumerate(pics):
    if ".bmp" not in el:
        pics.remove(el)
# print(len(pics))

def remove_neg(arr):
    for i, el in enumerate(arr):
        if el < 0:
            arr[i] = 0
    return arr


class ExampleApp(Frame):
    def __init__(self, master):
        Frame.__init__(self,master=None, height=576, width=720)
        self.x = self.y = 0
        self.canvas = Canvas(self,  cursor="cross", height=576, width=720)

        self.sbarv=Scrollbar(self, orient=VERTICAL)
        self.sbarh=Scrollbar(self, orient=HORIZONTAL)
        self.sbarv.config(command=self.canvas.yview)
        self.sbarh.config(command=self.canvas.xview)

        self.canvas.config(yscrollcommand=self.sbarv.set)
        self.canvas.config(xscrollcommand=self.sbarh.set)

        self.canvas.grid(row=0, column=0, sticky=N+S+E+W)
        self.sbarv.grid(row=0, column=1, stick=N+S)
        self.sbarh.grid(row=1, column=0, sticky=E+W)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.rect = None

        self.start_x = None
        self.start_y = None

        self.im = PIL.Image.open(path_base+pics[counter_global])
        self.arr = cv.imread(path_base+pics[counter_global])
        self.wazil,self.lard=self.im.size
        self.canvas.config(scrollregion=(0,0,self.wazil,self.lard))
        self.tk_im = ImageTk.PhotoImage(self.im)
        self.canvas.create_image(0,0,anchor="nw",image=self.tk_im)


    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        # print(self.start_x, self.start_y)

        # create rectangle if not yet exist
        if not self.rect:
            self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='red')

    def on_move_press(self, event):
        curX = self.canvas.canvasx(event.x)
        curY = self.canvas.canvasy(event.y)
        self.canvas.tag_raise(self.rect, "all")

        # w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        # if event.x > 0.9*w:
        #     self.canvas.xview_scroll(1, 'units')
        # elif event.x < 0.1*w:
        #     self.canvas.xview_scroll(-1, 'units')
        # if event.y > 0.9*h:
        #     self.canvas.yview_scroll(1, 'units')
        # elif event.y < 0.1*h:
        #     self.canvas.yview_scroll(-1, 'units')

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)

    def on_button_release(self, event):
        global counter_global, counter_help
        try:
            # print(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
            # print(self.arr.shape)
            arr = [self.start_y, self.start_x, self.canvas.canvasy(event.y), self.canvas.canvasx(event.x)]
            y1, x1, y2, x2 = remove_neg(arr)
            arr1 = self.arr[min(int(y1),int(y2)):max(int(y1),int(y2)),
                   min(int(x1), int(x2)):max(int(x1), int(x2))]
            print(arr1.shape)
            cv.imwrite(save_base.format(pics[counter_global].split(".")[0], counter_help % 5 + 1), arr1)
            counter_help += 1
            if counter_help % cuts_per_im == 0:
                counter_global += 1
                print(path_base + pics[counter_global])
                self.im = PIL.Image.open(path_base + pics[counter_global])
                self.arr = cv.imread(path_base + pics[counter_global])
                self.wazil, self.lard = self.im.size
                self.canvas.config(scrollregion=(0, 0, self.wazil, self.lard))
                self.tk_im = ImageTk.PhotoImage(self.im)
                self.canvas.create_image(0, 0, anchor="nw", image=self.tk_im)
        except:
            # counter_help -= 1
            # print(e)
            print("try once more")


if __name__ == "__main__":
    root=Tk()
    app = ExampleApp(root)
    app.pack()
    root.mainloop()