from tkinter import *
import numpy as np
from PIL import ImageGrab
from prediction import predict

window = Tk()
window.title("Handwritten Digit Recognition")
l1 = Label()

def MyProject():
    global l1

    widget = cv
    # setting coordinates of canvas
    x = window.winfo_rootx() + widget.winfo_x()
    y = window.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    # image is captured from canvas and resized to 28x28 pixels
    img = ImageGrab.grab().crop((x, y, x1, y1)).resize((28, 28))

    # convert rgb to grayscale img
    img = img.convert('L')

    # extract pixel matrix of img and convert it to vector of (1, 784)
    x = np.asarray(img)
    vec = np.zeros((1, 784))
    k = 0
    for i in range(28):
        for j in range(28):
            vec[0][k] = x[i][j]
            k +=1
    
    # loading thetas
    theta1 = np.loadtxt('theta1.txt')
    theta2 = np.loadtxt('theta2.txt')

    # calling predict function
    pred = predict(theta1, theta2, vec / 255)

    l1 = Label(window, text="Digit = " + str(pred[0]), font=('Arial', 20))
    l1.place(x=230, y=420)

lastx, lasty = None, None

# clears canvas
def clear_widget():
    global cv, l1
    cv.delete("all")
    l1.destroy()

# activates canvas
def event_activation(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y

# draw on canvas
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=25, fill='white', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y

# label
L1 = Label(window, text="Handwritten Digit Recognition", font=('Arial', 25), fg="blue")
L1.place(x=35, y=10)

# button to clear canvas
b1 = Button(window, text="1. Clear Canvas", font=('Arial', 15), bg='orange', fg='black', command=clear_widget)
b1.place(x=120, y=370)

# button to predict canvas
b2 = Button(window, text="2. Prediction", font=('Arial', 15), bg='white', fg='red', command=MyProject)
b2.place(x=320, y=370)

# setting properties of canvas
cv = Canvas(window, width=350, height=290, bg='black')
cv.place(x=120, y=70)

cv.bind('<Button-1>', event_activation)
window.geometry("600x500")
window.mainloop()