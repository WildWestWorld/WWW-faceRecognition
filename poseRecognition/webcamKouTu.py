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
                     model_complexity=1, #选择人体姿态关键点检测模型，0性能差但是快，2性能好但是慢，1最平均
                     smooth_landmarks=True, #是否平滑关键点
                     min_detection_confidence=0.5,  # 置信度阈值 0.7 比较好
                     min_tracking_confidence=0.5)  # 追踪阈值 默认就好

# 导入绘图函数
mpDraw = mp.solutions.drawing_utils


# # 处理单帧的函数

# 处理帧函数
def process_frame(img):
    # 水平镜像翻转图像，使图中左右手与真实左右手对应
    # 参数 1：水平翻转，0：竖直翻转，-1：水平和竖直都翻转
    img = cv2.flip(img, 1)
    # BGR转RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将RGB图像输入模型，获取预测结果
    results = pose.process(img_RGB)

    # if results.pose_landmarks:  # 如果有检测到手  results.multi_hand_landmarks里面有值
    #     # 遍历每一只检测出的手
    #     for i in range(len(results.pose_landmarks)):
    pose_42 = results.pose_landmarks  # 获取该手的所有关键点坐标
    #         # hand_21 :
    #         # landmark {
    #         #   x: 0.5877419710159302
    #         #   y: 0.6618870496749878
    #         #   z: -0.015523000620305538
    #         # }
    #         # mpDraw.draw_landmarks(要画的图，手的所有关键点坐标，手部关键点模型 用什么连线)
    mpDraw.draw_landmarks(img, pose_42, mp_pose.POSE_CONNECTIONS)  # 可视化

    return img




import cv2
import time

# 获取摄像头，传入0表示获取系统默认摄像头
cap = cv2.VideoCapture(0)

# 打开cap
cap.open(0)

# 无限循环，直到break被触发
while cap.isOpened():
    # 获取画面
    success, frame = cap.read()
    if not success:
        print('Error')
        break
    start_time = time.time()

    ## !!!处理帧函数
    frame = process_frame(frame)

    # 展示处理后的三通道图像
    cv2.imshow('my_window', frame)

    if cv2.waitKey(1) in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
        break

# 关闭摄像头
cap.release()

# 关闭图像窗口
cv2.destroyAllWindows()
