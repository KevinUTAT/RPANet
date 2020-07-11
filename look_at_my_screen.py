from PIL import ImageGrab
import numpy as np
import cv2
import pyautogui
import win32api


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
    while(True):
        img = ImageGrab.grab(bbox=(screen_x[0],screen_y[0],screen_x[1],screen_y[1]),\
            all_screens=True) #bbox specifies specific region (bbox= x,y,width,height)
        img_np = np.array(img)
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imshow("Debug", frame)
        cv2.waitKey(25)
    cv2.destroyAllWindows()