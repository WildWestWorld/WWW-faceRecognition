# -*- coding: utf-8 -*-


import numpy as np
from PIL import ImageGrab
import cv2
import time
import directkeys
import grabscreen
import getkeys
import os
import numpy as np
import win32gui, win32ui, win32con, win32api
import mss

wait_time = 5
L_t = 3
file_name = 'training_data_2_1.npy'
sct = mss.mss()
monitor = {'left': 290, 'top': 0, 'width': 960, 'height': 960}


if os.path.isfile(file_name):
    print("file exists , loading previous data")
    training_data = list(np.load(file_name,allow_pickle=True))
else:
    print("file don't exists , create new one")
    training_data = []


    for i in list(range(5))[::-1]:
        print(i+1)
        time.sleep(1)
last_time = time.time()

while True:
    keys=getkeys.key_check()
    print(keys)
    output_key = getkeys.get_key(keys)#按键收集
    print(output_key)

    # if output_key  == [1, 1, 1, 1,1, 1, 1, 1, 1, 1]:
    #     print(len(training_data))
    #     np.save(file_name,training_data)
    #     break


    img= sct.grab(monitor=monitor);
    img = np.array(img)

    screen_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#灰度图像收集

    screen_reshape = cv2.resize(screen_gray,(96,86))

    if output_key[9] !=1:
        training_data.append([screen_reshape,output_key])
    
    if len(training_data) % 500 == 0:
        print(len(training_data))
        np.save(file_name,training_data)
    
    cv2.imshow('window1',screen_gray)
    #cv2.imshow('window3',printscreen)
    #cv2.imshow('window2',screen_reshape)
    
    #测试时间用
    print('FPS {} '.format(1/(time.time()-last_time)))
    last_time = time.time()
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(len(training_data))
        np.save(file_name, training_data)
        break
cv2.destroyAllWindows()
