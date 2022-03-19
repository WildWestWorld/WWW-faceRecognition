# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:29:20 2020

@author: analoganddigital   ( GitHub )
"""

import numpy as np
from grabscreen import grab_screen
import cv2
import time
import mss
import directkeys
from Alexnet import alexnet2
from getkeys import key_check
import pyautogui

import random

WIDTH = 96
HEIGHT = 86
LR = 1e-3
EPOCHS = 20
MODEL_NAME = 'model/py-sekiro-{}-{}-{}-epochs.model'.format(LR,'alexnetv2',EPOCHS)
sct = mss.mss()
monitor = {'left': 290, 'top': 0, 'width': 960, 'height': 960}
#w j m k none

w = [1,0,0,0,0,0,0,0,0,0]
a = [0,1,0,0,0,0,0,0,0,0]
s = [0,0,1,0,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0,0]
j = [0,0,0,0,1,0,0,0,0,0]
k = [0,0,0,0,0,1,0,0,0,0]
l = [0,0,0,0,0,0,1,0,0,0]
u = [0,0,0,0,0,0,0,1,0,0]
i = [0,0,0,0,0,0,0,0,1,0]

wj =[1,0,0,0,1,0,0,0,0,0]
wu =[1,0,0,0,0,0,0,1,0,0]
wi =[1,0,0,0,0,0,0,0,1,0]

sj =[0,0,1,0,1,0,0,0,0,0]
su =[0,0,1,0,0,0,0,1,0,0]
si =[0,0,1,0,0,0,0,0,1,0]



model = alexnet2(WIDTH, HEIGHT, LR, output = 10)
model.load(MODEL_NAME)



def main():
    last_time = time.time()
    for i in list(range(5))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while(True):
        
        if not paused:
            # 800x600 windowed mode
            screen = sct.grab(monitor=monitor);
            screen = np.array(screen)

            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (WIDTH,HEIGHT))

            prediction = model.predict([screen.reshape(WIDTH,HEIGHT,1)])[0]
            print(prediction)
            
            print("np后的预测:",np.argmax(prediction))

            if np.argmax(prediction) == np.argmax(w):
                print("w")
                pyautogui.press('w')
            elif np.argmax(prediction) == np.argmax(a):
                print("a")
                pyautogui.press('a')
            elif np.argmax(prediction) == np.argmax(s):
                print("s")
                pyautogui.press('s')
            elif np.argmax(prediction) == np.argmax(d):
                print("d")
                pyautogui.press('d')

            elif np.argmax(prediction) == np.argmax(j):
                print("j")
                pyautogui.press('j')
            elif np.argmax(prediction) == np.argmax(k):
                print("k")
                pyautogui.press('k')
            elif np.argmax(prediction) == np.argmax(l):
                print("l")
                pyautogui.press('l')

            elif np.argmax(prediction) == np.argmax(wj):
                print("wj")
                pyautogui.hotkey('w','j');
            elif np.argmax(prediction) == np.argmax(wu):
                print("wu")
                pyautogui.hotkey('w','u');
            elif np.argmax(prediction) == np.argmax(wi):
                print("wi")
                pyautogui.hotkey('w','i');
            elif np.argmax(prediction) == np.argmax(sj):
                print("sj")
                pyautogui.hotkey('s','j');
            elif np.argmax(prediction) == np.argmax(su):
                print("su")
                pyautogui.hotkey('s','u');
            elif np.argmax(prediction) == np.argmax(si):
                print("si")
                pyautogui.hotkey('s','i');
        keys = key_check()


        #如果按T就安亭
        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                '''
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                '''
                time.sleep(1)
        if 'Y' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                '''
                directkeys.ReleaseKey(J)
                directkeys.ReleaseKey(W)
                directkeys.ReleaseKey(M)
                directkeys.ReleaseKey(K)
                '''
                time.sleep(1)
                break

main()  