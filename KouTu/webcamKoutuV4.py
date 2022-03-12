

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
import numpy as np

# 导入solution
mp_pose = mp.solutions.selfie_segmentation
# 导入模型
SS = mp_pose.SelfieSegmentation(model_selection=0  #选择几号模型
                                )

# 导入绘图函数
mpDraw = mp.solutions.drawing_utils


# # 处理单帧的函数

# 处理帧函数
def process_frame(img,videoFrame):
    # 水平镜像翻转图像，使图中左右手与真实左右手对应
    # 参数 1：水平翻转，0：竖直翻转，-1：水平和竖直都翻转
    img = cv2.flip(img, 1)


    bgc_img=videoFrame
    # bgc_img=img_resize(bgc_img)

    #图片的Y轴是向下的
    #先把背景图里面的部分图块切除
    #bgc_img的高度/宽度
    # bgc_img_height=bgc_img.shape[0]
    # bgc_img_width=bgc_img.shape[1]

    #原图的高度/宽度
    img_height=img.shape[0]
    img_width=img.shape[1]

    bgc_img=cv2.resize(bgc_img,(int(img_width),(int(img_height))))
    # BGR转RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将RGB图像输入模型，获取预测结果
    results = SS.process(img_RGB)


    mask = results.segmentation_mask
    #将结果里面的数都变成无符号整形，作用是让那些小数都消失，让他里面的书都变成非0即1的
    #此时的mask是一张平面图，而且是单通道的我们需要把他变成三通道的


    #把单通道变成三通道了，就是把3个mask堆叠到一块
    mask_3=np.stack((mask,mask,mask),axis=-1)
    #把mask_3 里面的数变成逻辑符号True/False
    #后面的np.where 会根据true或者false来
    mask_3=mask_3>0.02


#开始更换背景颜色
    #新建一张新的图片
    # mask_bgc=[0,200,0]
    # #搞一个和imgshape形状的全0序列
    # img_without_bgc=np.zeros(img.shape,dtype=np.uint8)
    # #把全0的序列数据替换为mask_bgc
    # img_without_bgc[0:]=mask_bgc
    #
    # #np.where np中的三目运算符
    # #mask若为true 该位置就是img
    # #若为false 该位置就是img_without_bgc
    # img_without_bgc=np.where(mask_3,img,img_without_bgc)
    #
    # #支线：获取扣除图像后的背景
    # img_without_person=np.where(~mask_3,img,img_without_bgc)

#更换背景图片
    #背景图片的大小要大于原图片


    # Bottom=bgc_img_height
    # Top=bgc_img_height-img_height;
    #
    # #因为要放在中间所以我们得/2
    # Left=int(bgc_img_width/2-img_width/2)
    # Right=Left+img_width;
    #
    # #在背景图中抠出图像
    # bgc_img_KT=bgc_img[Top:Bottom,Left:Right,0:];

    # #np.where np中的三目运算符
    # #mask若为true 该位置就是img
    # #若为false 该位置就是img_without_bgc
    bgc_img_KT_addPerson=np.where(mask_3,img,bgc_img)

    #把加上人像后的抠图放回到原图的位置
    bgc_img=bgc_img_KT_addPerson;


    return bgc_img

# #缩放图片
# def img_resize(image):
#     height, width = image.shape[0], image.shape[1]
#     # 设置新的图片分辨率框架
#     width_new = 960
#     height_new = 720
#     img_new=cv2.resize(image,(int(width* height_new / height),(int(height * width_new / width))))
#
#     return img_new


import cv2
import time


# 获取摄像头，传入0表示获取系统默认摄像头
videoPath="test3.flv"

cap = cv2.VideoCapture(0)
video = cv2.VideoCapture(videoPath)
frame_counter = 0


# 打开cap
cap.open(0)


# 无限循环，直到break被触发
while cap.isOpened():
    # 获取画面
    success, frame = cap.read()
    ret, videoFrame = video.read()

    totalFrame=int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_counter += 1
    if frame_counter == int(totalFrame*0.98):
        frame_counter = 0

        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ret, videoFrameRest = video.read()
        videoFrame=videoFrameRest


    if not success:
        print('Error')
        break
    start_time = time.time()

    ## !!!处理帧函数
    frame = process_frame(frame,videoFrame)

    # 展示处理后的三通道图像
    cv2.imshow('my_window', frame)

    if cv2.waitKey(1) in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
        break

# 关闭摄像头
cap.release()
video.release()
# 关闭图像窗口
cv2.destroyAllWindows()






