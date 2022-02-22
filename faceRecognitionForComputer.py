import cv2
import mss
import numpy
# 参考文献 https://zhuanlan.zhihu.com/p/66368987
sct = mss.mss()

monitor = {'left': 290, 'top': 0, 'width': 960, 'height': 640}
while True:
    img = sct.grab(monitor=monitor);

    imgArr = numpy.array(img)

    # cv2.imshow("test", imgArr)

    # 导入人脸级联分类器引擎，'.xml'文件里包含训练出来的人脸特征
    #简单来说就是模型库 后面跟的是模型的名称
    face_engine = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    eye_engine = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


    smile_engine = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    # 用人脸级联分类器引擎进行人脸识别，返回的faces为人脸坐标列表，1.3是放大比例，5是重复识别次数
    faces = face_engine.detectMultiScale(imgArr, scaleFactor=1.3, minNeighbors=5)

    # 对每一张脸，进行如下操作
    for (x, y, w, h) in faces:
        # 画出人脸框，   (255, 0, 0)蓝色（BGR色彩体系），画笔宽度为2
        imgArr = cv2.rectangle(imgArr, (x, y), (x + w, y + h), (255, 0, 0), 2)

        #把框选出的人脸框变成图片 目的：让眼睛检测在人脸框中检测 节省计算资源
        faces_area=imgArr[y:y+h,x:x+w]
        # 用人脸级联分类器引擎进行人脸识别，返回的faces为人脸坐标列表，1.3是放大比例，5是重复识别次数
        eyes = eye_engine.detectMultiScale(faces_area, scaleFactor=1.3, minNeighbors=15)
        # 画出人眼框，   (255, 0, 0)绿色（BGR色彩体系），画笔宽度为1
        for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(faces_area,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        smile = smile_engine.detectMultiScale(faces_area, scaleFactor=1.3, minNeighbors=15)

        for (sx,sy,sw,sh) in smile:
                cv2.rectangle(faces_area,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
    # imgArr = numpy.array(img)
    # 在"img2"窗口中展示效果图
    cv2.imshow('img2', imgArr)

    pushKeyboard = cv2.waitKey(1);

    if (pushKeyboard % 256 == 27):
     cv2.destroyAllWindows();
     exit("停止截屏")
