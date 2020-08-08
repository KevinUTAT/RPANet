from PIL import ImageGrab
from PIL import Image
import numpy as np
import cv2
import pyautogui
import win32api
import time
from mss import mss



def get_screen_regin():
    pos_x = []
    pos_y = []
    while True:
        state_left = win32api.GetKeyState(0x01)
        x, y = pyautogui.position()
        pos_str = "X: " + str(x).rjust(40) + "Y: " + str(y).rjust(40)
        print(pos_str, end='')
        print('\b' * len(pos_str), end='', flush=True)
        a = win32api.GetKeyState(0x01)
        if a != state_left and a < 0:
            pos_x.append(x)
            pos_y.append(y)
        if len(pos_x) >= 2:
            return pos_x, pos_y


if __name__ == '__main__': 
    screen_x, screen_y = get_screen_regin()
    print(screen_x[0],screen_y[0],screen_x[1],screen_y[1])
    window = {'top': screen_y[0], 'left': screen_x[0], 'width': (screen_x[1] - screen_x[0]), 'height': (screen_y[1] - screen_y[0])}
    while(True):
        start_time = time.time()
        # img = ImageGrab.grab(bbox=(screen_x[0],screen_y[0],screen_x[1],screen_y[1]),\
        #     all_screens=True) #bbox specifies specific region (bbox= x,y,width,height)
        # img_np = np.array(img)
        # frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cap = mss().grab(window)
        img = Image.frombytes("RGB", (cap.width, cap.height), cap.rgb)
        print("Time0: ", time.time() - start_time)
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        print("Time1: ", time.time() - start_time)
        cv2.imshow("Debug", frame)
        cv2.waitKey(25)
    cv2.destroyAllWindows()