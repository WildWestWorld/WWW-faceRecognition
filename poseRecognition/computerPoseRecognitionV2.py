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
mp_pose = mp.solutions.pose
# 导入模型
pose = mp_pose.Pose(static_image_mode=False,  # 是静态图片还是连续视频帧
                     model_complexity=2, #选择人体姿态关键点检测模型，0性能差但是快，2性能好但是慢，1最平均
                     smooth_landmarks=True, #是否平滑关键点
                     enable_segmentation=True, #是否人体抠图
                     min_detection_confidence=0.7,  # 置信度阈值 0.7 比较好
                     min_tracking_confidence=0.5)  # 追踪阈值 默认就好

# 导入绘图函数
mpDraw = mp.solutions.drawing_utils


# # 处理单帧的函数

# 处理帧函数
def process_frame(img):

    # BGR转RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将RGB图像输入模型，获取预测结果
    results = pose.process(img)

    if results.pose_landmarks:  # 如果有检测到手  results.multi_hand_landmarks里面有值

        pose_42 = results.pose_landmarks  # 获取该手的所有关键点坐标

        mpDraw.draw_landmarks(img, pose_42, mp_pose.POSE_CONNECTIONS)  # 可视化

    #人体抠图
    #mask代表每一个像素是不是对应人体的概率
    mask =results.segmentation_mask
    #让mask里面概率大于0.5的变成1 ，小于等于0.5的变成0
    mask =mask>0.5;
    return img


# # 调用摄像头获取每帧（模板）

import cv2
import time
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


    start_time = time.time()

    ## !!!处理帧函数
    frame = process_frame(imgArr)

    # 展示处理后的三通道图像
    cv2.imshow('my_window', frame)

    if cv2.waitKey(1) in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
        break


# 关闭图像窗口
cv2.destroyAllWindows()
