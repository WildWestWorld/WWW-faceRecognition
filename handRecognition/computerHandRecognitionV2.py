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
                       min_detection_confidence=0.5,  # 置信度阈值 0.7 比较好
                       min_tracking_confidence=0.5)  # 追踪阈值 默认就好

# 导入绘图函数
mpDraw = mp.solutions.drawing_utils


# # 处理单帧的函数

# 处理帧函数
def process_frame(img):
    # 记录该帧开始处理的时间
    # 用于FPS的计算
    start_time = time.time()

    # 获取图像宽高
    h, w = img.shape[0], img.shape[1]
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    # BGR转RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将RGB图像输入模型，获取预测结果
    results = hands.process(img_RGB)

    # 识别左右手，和他的置信度
    # print(results.multi_handedness)

    # [classification {
    #   index: 1
    #   score: 0.9420602321624756
    #   label: "Right"
    # }

    # 手所在的位置
    # print(results.multi_hand_landmarks)
    # landmark
    # {
    #     x: 0.4599137306213379
    #     y: 1.0057446956634521
    #     z: -0.05328677222132683
    # }

    if results.multi_hand_landmarks:  # 如果有检测到手

        handness_str = ''
        index_finger_tip_str = ''

        # hand_idx 就是i
        for hand_idx in range(len(results.multi_hand_landmarks)):

            # 获取该手的21个关键点坐标
            hand_21 = results.multi_hand_landmarks[hand_idx]

            # 画线
            # 可视化关键点及骨架连线
            # mpDraw.draw_landmarks(要画的图，手的所有关键点坐标，手部关键点模型 用什么连线)
            mpDraw.draw_landmarks(img, hand_21, mp_hands.HAND_CONNECTIONS)

            # 记录左右手信息
            # results.multi_handedness[hand_idx].classification[0].label =  记录左手还是右手

            temp_handness = results.multi_handedness[hand_idx].classification[0].label

            if temp_handness=="Left":
                temp_handness="Right"
            elif temp_handness=="Right":
                temp_handness = "Left"

            # >>> "{0} {1}".format("hello", "world")  # 设置指定位置
            # 'hello world'

            # hand_idx = i
            # results.multi_handedness[hand_idx].classification[0].label =  记录左手还是右手
            handness_str += '{0}:{1} '.format(hand_idx, temp_handness)

            # 获取手腕根部深度坐标
            cz0 = hand_21.landmark[0].z

            for i in range(21):  # 遍历该手的21个关键点

                # 获取3D坐标
                # 图片中的高度和宽度
                cx = int(hand_21.landmark[i].x * w)
                cy = int(hand_21.landmark[i].y * h)
                # cz当前检测的z坐标
                cz = hand_21.landmark[i].z
                # cz0掌根的Z轴，cz当前检测的z坐标
                depth_z = cz0 - cz

                # 用圆的半径反映深度大小
                radius = max(int(6 * (1 + depth_z * 5)), 0)

                if i == 0:  # 手腕
                    # cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
                    # 图片，坐标，圆的半径，颜色，线的粗细
                    img = cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)

                if i == 4:  # 拇指指尖
                    img = cv2.circle(img, (cx, cy), radius, (161 , 47 , 47), -1)
                if i == 8:  # 食指指尖
                    img = cv2.circle(img, (cx, cy), radius, (277 , 23 , 13), -1)
                    # 将相对于手腕的深度距离显示在画面中
                    index_finger_tip_str += '{}:{:.2f} '.format(hand_idx, depth_z)
                if i == 12:  # 中指指尖
                    img = cv2.circle(img, (cx, cy), radius, (20 , 0 , 28), -1)

                if i == 16:  # 无名指指指尖
                    img = cv2.circle(img, (cx, cy), radius, (34  , 139  , 34), -1)

                if i == 20:  # 小拇指指尖
                    img = cv2.circle(img, (cx, cy), radius, (244  , 208  , 0), -1)

                if i in [1, 5, 9, 13, 17]:  # 指根
                    img = cv2.circle(img, (cx, cy), radius, (16, 144, 247), -1)
                if i in [2, 6, 10, 14, 18]:  # 第一指节
                    img = cv2.circle(img, (cx, cy), radius, (1, 240, 255), -1)
                if i in [3, 7, 11, 15, 19]:  # 第二指节
                    img = cv2.circle(img, (cx, cy), radius, (140, 47, 240), -1)
                # if i in [4, 12, 16, 20]:  # 指尖（除食指指尖）
                #     img = cv2.circle(img, (cx, cy), radius, (223, 155, 60), -1)

        scaler = 1

        # 在图像上写数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        #            #hand_idx = i
        # results.multi_handedness[hand_idx].classification[0].label =  记录左手还是右手
        # handness_str += '{0}:{1} '.format(hand_idx, temp_handness)

        img = cv2.putText(img, handness_str, (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler,
                          (255, 0, 255), 2 * scaler)

        img = cv2.putText(img, index_finger_tip_str, (25 * scaler, 150* scaler), cv2.FONT_HERSHEY_SIMPLEX,
                          1.25 * scaler, (255, 0, 255), 2 * scaler)

        # 记录该帧处理完毕的时间
        # 下面都是为了把FPS写在窗口上
        end_time = time.time()
        # 计算每秒处理图像帧数FPS
        FPS = 1 / (end_time - start_time)

        scaler = 1
        # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        img = cv2.putText(img, 'FPS  ' + str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
                          1.25 * scaler, (255, 0, 255), 2 * scaler)
    return img


# # 调用摄像头获取每帧（模板）

# 调用摄像头逐帧实时处理模板
# 不需修改任何代码，只需修改process_frame函数即可
# 同济子豪兄 2021-7-8

# 导入opencv-python
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

    ## !!!处理帧函数
    frame = process_frame(imgArr)




    # 展示处理后的三通道图像
    cv2.imshow('my_window', frame)

    pushKeyboard = cv2.waitKey(1);

    if (pushKeyboard % 256 == 27):
        cv2.destroyAllWindows();
        exit("停止截屏")
        break;
