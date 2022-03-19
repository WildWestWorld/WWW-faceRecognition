
import win32api
import time

keyList = ["\b"]
#这些符号放入到keyList数组中
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        #ord 将字母转化成10进制
        #看下这些按键是否可以被win32api调用，如果可以就把这些按键放进去
        #也就是说你按下哪个键，哪个键就会被返回，而且可以返回多个键
        if win32api.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys

def get_key(keys):
    #W,J,M,K,R,none
    output = [0,0,0,0,0,0,0,0,0,0]
    if 'W' in keys:
        output[0] = 1
    elif 'A' in keys:
        output[1] = 1
    elif 'S' in keys:
        output[2] = 1
    elif 'D' in keys:
        output[3] = 1 
    elif 'J' in keys:
        output[4] = 1
    elif 'K' in keys:
        output[5] = 1
    elif 'L' in keys:
        output[6] = 1
    elif 'U' in keys:
        output[7] = 1
    elif 'I' in keys:
        output[8] = 1

    elif '8' in keys:#停止记录操作
        output = [1, 1, 1, 1,1, 1, 1, 1, 1, 1]

    else:
        output[9] = 1
        
    return output

