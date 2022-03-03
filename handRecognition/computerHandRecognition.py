#!/usr/bin/env python
# coding: utf-8

# # 导入工具包

# opencv-python
import cv2
# mediapipe人工智能工具包
import mediapipe as mp
# 进度条库
from tqdm import tqdm
# 时间库
import time

# # 导入手部关键点模型

# 导入solution


mp_hands = mp.solutions.hands
# 导入模型
hands = mp_hands.Hands(static_image_mode=False,  # 是静态图片还是连续视频帧
                       max_num_hands=4,  # 最多检测几只手
                       min_detection_confidence=0.5,  # 置信度阈值
                       min_tracking_confidence=0.5)  # 追踪阈值

# 导入绘图函数
mpDraw = mp.solutions.drawing_utils


# # 处理单帧的函数

# 处理帧函数
def process_frame(img):
    # BGR转RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # img_RGB=img.transpose((0, 1, 2))
    # 将RGB图像输入模型，获取预测结果
    results = hands.process(img_RGB)

    if results.multi_hand_landmarks:  # 如果有检测到手
        # 遍历每一只检测出的手
        for hand_idx in range(len(results.multi_hand_landmarks)):
            hand_21 = results.multi_hand_landmarks[hand_idx]  # 获取该手的所有关键点坐标
            mpDraw.draw_landmarks(img, hand_21, mp_hands.HAND_CONNECTIONS)  # 可视化

    return img


# # 调用摄像头获取每帧（模板）

# 调用摄像头逐帧实时处理模板
# 不需修改任何代码，只需修改process_frame函数即可
# 同济子豪兄 2021-7-8

# 导入opencv-python
import cv2
import time
import mss
import numpy

sct = mss.mss()

monitor = {'left': 290, 'top': 0, 'width': 960, 'height': 960}
# 无限循环，直到break被触发
while True:
    # 获取画面
    img = sct.grab(monitor=monitor);

    imgArr = numpy.array(img)

    ## !!!处理帧函数
    frame = process_frame(imgArr)

    # 展示处理后的三通道图像
    cv2.imshow('my_window', frame)

    pushKeyboard = cv2.waitKey(1);

    if (pushKeyboard % 256 == 27):
        cv2.destroyAllWindows();
        exit("停止截屏")
        break;

