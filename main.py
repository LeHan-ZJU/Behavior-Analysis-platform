import sys
from OboardCamDisp import Ui_MainWindow
from PIL import Image
from collections import deque
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QDateTimeEdit
from PyQt5.QtCore import QTimer, QCoreApplication, QPoint, QDateTime
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QCursor
import qdarkstyle
import numpy as np
import cv2
import qimage2ndarray
import time
import datetime
import os
import csv
import torch
import torchvision
from RatNet1 import PoseDetection, predict_img, initNet, noDetection, draw_relation


def script_method(fn, _rcb=None):
    return fn


def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj
import torch.jit
torch.jit.script_method = script_method
torch.jit.script = script


def CalculateLength(mask):
    mask_erode = cv2.erode(mask, None, iterations=2)
    return mask_erode

def CalculateForegrounds(Bg, img, kernel, pts, ChangeBgmFlag, AreaThreshold, GrayThreshold, signalLamp, KeyPointsFlag):
    img0 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    Bg0 = cv2.cvtColor(Bg, cv2.COLOR_RGB2GRAY)
    img_Guassian = cv2.blur(img0, (5, 5))  # .astype(np.float32)
    Bg_Guassian = cv2.blur(Bg0, (5, 5))

    if ChangeBgmFlag==0:
        Foreground = cv2.subtract(img_Guassian, Bg_Guassian)
    else:
        Foreground = cv2.subtract(Bg_Guassian, img_Guassian)

    Foreground = cv2.morphologyEx(Foreground, cv2.MORPH_OPEN, kernel)
    retval, Foreground = cv2.threshold(Foreground, GrayThreshold, 255, cv2.THRESH_BINARY)

    Foreground = CalculateLength(Foreground)
    contours2, hierarchy2 = cv2.findContours(Foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect=0

    for c in contours2:
        perimeter = cv2.arcLength(c, True)
        if perimeter > AreaThreshold:
            rect = cv2.minAreaRect(c) 
            box = np.int0(cv2.boxPoints(rect))
            if KeyPointsFlag == 0:
                cv2.drawContours(img, [box], -1, (0, 255, 0), 2)   
            CenterPoint = np.int0(rect[0])
            mm = cv2.moments(c)
            m00 = mm['m00']
            m10 = mm['m10']
            m01 = mm['m01']
            if m00:
                cx = np.int(m10 / m00)
                cy = np.int(m01 / m00)
                cv2.circle(img, (cx, cy), 2, (255, 0, 0), 4) 
            if KeyPointsFlag == 0:
                cv2.drawContours(img, contours2, -1, (0, 0, 255), 1)  
    return img, rect

def JudgeDeriection(pts):
    len = pts.shape
    a = pts[len[0]-5]
    b = pts[len[0]-2]
    c = pts[len[0]-1]
    # S(P1, P2, P3) = | y1 y2 y3 |= (x1 - x3) * (y2 - y3) - (y1 - y3) * (x2 - x3)
    s = (a[0] - c[0])*(b[1]-c[1]) - (a[1] - c[1])*(b[0]-c[0])  
    return s

def BackgroundModeling(frame, kernel, fgbg):
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        if perimeter > 188:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def CalculateSpeed(cx1, cy1, cx2, cy2, t1, t2, DisScale):
    speed = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    
    speed = speed//np.abs(t2-t1)
    speed = speed/DisScale  
    return speed


def get_dis(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_speed(center_list, DisScale, frame_rate, window_len=10):
    speed_list = []
    for i in range(len(center_list) - 10):
        cur_dis = 0
        for j in range(window_len):
            cur_dis += get_dis(center_list[i+j], center_list[i+j+1])
        speed_list.append((cur_dis * frame_rate) / (DisScale * 10))  # 10frame*3.4pix/s
    return speed_list

def CropCircleArea(Image, mid_x, mid_y, radius):
    for i in range(Image.shape[0]):
        for j in range(Image.shape[1]):
            if np.sqrt((j - mid_x) ** 2 + (i-mid_y) ** 2) > radius:
                Image[i, j, :] = 0
    return Image

def CropCircleArea2(Image, mid_x, mid_y, radius):
    img = Image - Image
    cv2.circle(img, (mid_x, mid_y), radius, (255, 255, 255), -1)
    img = np.array(img[:,:,0])/255
    for i in range(3):
        Image[:,:,i] = Image[:,:,i] * img
    return Image


def DrawcircleArea(Image, circle_start_pos, circle_end_pos, circle_start_x, circle_start_y, circle_end_x, circle_end_y):
    if circle_start_pos != None:
        cv2.circle(Image, (circle_start_x, circle_start_y), 2, (100, 160, 60), 1)
    if circle_end_pos != None:
        cv2.circle(Image, (circle_end_x, circle_end_y), 2, (100, 160, 60), 1)
        radius = int(np.sqrt((circle_start_x - circle_end_x) ** 2 + (
                    circle_start_y - circle_end_y) ** 2) / 2)
        mid_x = int((circle_start_x + circle_end_x) / 2)
        mid_y = int((circle_start_y + circle_end_y) / 2)
        cv2.circle(Image, (mid_x, mid_y), radius, (100, 160, 60), 1)
        Image = CropCircleArea2(Image, mid_x, mid_y, radius)
        return Image, radius, mid_x, mid_y
    else:
        return Image, 0, 0, 0

def DrawRectangleArea(Image, rectangle_start_pos, rectangle_end_pos, rectangle_start_x, rectangle_start_y, rectangle_end_x, rectangle_end_y):
    if rectangle_start_pos != None:
        cv2.circle(Image, (rectangle_start_x, rectangle_start_y), 2, (255, 0, 155), 1)
    if rectangle_end_pos != None:
        cv2.circle(Image, (rectangle_end_x, rectangle_end_y), 2, (255, 0, 155), 1)
        cv2.rectangle(Image, (rectangle_start_x, rectangle_start_y),
                      (rectangle_end_x, rectangle_end_y), (100, 160, 60), 1)
    return Image


class CamShow(QMainWindow,Ui_MainWindow):
    def __del__(self):
        try:
            self.camera.release() 
        except:
            return
    def __init__(self,parent=None):
        super(CamShow,self).__init__(parent)
        self.setupUi(self)
        self.PrepSliders()
        self.PrepParameters()
        self.PrepWidgets()
        self.PrepCameraParameters()
        self.CallBackFunctions()
        self.Timer=QTimer()
        self.Timer.timeout.connect(self.TimerOutFun)

    def PrepSliders(self):
        self.ExpTimeSld.valueChanged.connect(self.ExpTimeSpB.setValue)
        self.ExpTimeSpB.valueChanged.connect(self.ExpTimeSld.setValue)
        self.GainSld.valueChanged.connect(self.GainSpB.setValue)
        self.GainSpB.valueChanged.connect(self.GainSld.setValue)
        self.BrightSld.valueChanged.connect(self.BrightSpB.setValue)
        self.BrightSpB.valueChanged.connect(self.BrightSld.setValue)
        self.ContrastSld.valueChanged.connect(self.ContrastSpB.setValue)
        self.ContrastSpB.valueChanged.connect(self.ContrastSld.setValue)
        self.CameraNOSld.valueChanged.connect(self.CameraNOSpB.setValue)
        self.CameraNOSpB.valueChanged.connect(self.CameraNOSld.setValue)
        self.ThrSld.valueChanged.connect(self.ThrSpB.setValue)
        self.ThrSpB.valueChanged.connect(self.ThrSld.setValue)
        self.GrayThrSld.valueChanged.connect(self.GrayThrSpB.setValue)
        self.GrayThrSpB.valueChanged.connect(self.GrayThrSld.setValue)
        self.ScaleSld.valueChanged.connect(self.ScaleSpB.setValue)
        self.ScaleSpB.valueChanged.connect(self.ScaleSld.setValue)
        self.SpeedThrSld.valueChanged.connect(self.SpeedThrSpB.setValue)
        self.SpeedThrSpB.valueChanged.connect(self.SpeedThrSld.setValue)
        self.RatioLengthSpB.valueChanged.connect(self.RatioLengthSpB.setValue)

    def PrepWidgets(self):
        self.PrepCamera()
        self.StopBt.setEnabled(False)
        self.RecordBt.setEnabled(False)
        self.ExpTimeSld.setEnabled(False)
        self.ExpTimeSpB.setEnabled(False)
        self.GainSld.setEnabled(False)
        self.GainSpB.setEnabled(False)
        self.BrightSld.setEnabled(False)
        self.BrightSpB.setEnabled(False)
        self.ContrastSld.setEnabled(False)
        self.ContrastSpB.setEnabled(False)
        self.DetectBt.setEnabled(False)
        self.KeypointsBt.setEnabled(False)
        self.CameraNOSpB.setEnabled(True)
        self.CameraNOSld.setEnabled(True)

    def PrepParameters(self):
        self.RecordFlag = 0
        self.DetectFlag = 0
        self.KeyPointsFlag = 0
        self.CameraNum = 1
        self.ChangeBgmFlag = 0        
        self.AreaThreshold = 188
        self.GrayThreshold = 15
        self.RecordPath='./Saved-test/'
        self.pts = []  
        self.SpeedAll=[]
        self.signalLamp = 0     
        self.DisScale = 2.44    
        self.LogPath=self.RecordPath  
        self.LogCSVPath = self.RecordPath
        self.FilePathLE.setText(self.RecordPath)
        self.frame_num = 0
        self.Image_num = 0
        self.frame_rate = 30
        self.speed = 0
        self.SpeedThreshold = 5
        self.cx1 = 0
        self.cy1 = 0
        self.time1 = 0
        # self.time2
        self.CameraNOSpB.setValue(self.CameraNum)
        self.SetCameraNum()
        self.resetLabel=0  
        self.start_pos = None
        self.end_pos = None
        self.start_x, self.start_y, self.end_x, self.end_y = 0, 0, 0, 0
        self.ratioLineLabel = 0
        self.rectangle_start_pos = None  
        self.rectangle_end_pos = None
        self.rectangle_start_x, self.rectangle_start_y, self.rectangle_end_x, self.rectangle_end_y = 0, 0, 0, 0
        self.rectangleLabel = 0
        self.circle_start_pos = None  
        self.circle_end_pos = None
        self.circle_start_x, self.circle_start_y, self.circle_end_x, self.circle_end_y = 0, 0, 0, 0
        self.circleLabel = 0
        self.Angle = 0
        self.keyp_frame_num = 0
        self.keyPoints = np.zeros([2, 8])
        self.keyPoints_all = []  
        self.relation = [[1, 5], [1, 8], [1, 7], [2, 5], [2, 7], [2, 8], [3, 7], [3, 6], [4, 6], [4, 7], [5, 8], [7, 8]]


    def mousePressEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton and self.start_pos == None and self.resetLabel==2:
            self.start_pos = event.pos()
            self.start_x = int((self.start_pos.x() - 305) * 640 / 960)
            self.start_y = int((self.start_pos.y() - 60) * 480 / 720)
            self.resetLabel = 1
        elif event.buttons() == QtCore.Qt.LeftButton and self.start_pos != None and self.end_pos == None and self.resetLabel==1:
            self.end_pos = event.pos()
            self.end_x = int((self.end_pos.x() - 305) * 640 / 960)
            self.end_y = int((self.end_pos.y() - 60) * 480 / 720)
            self.resetLabel = 0
        elif event.buttons() == QtCore.Qt.LeftButton and self.rectangle_start_pos == None and self.rectangleLabel==1:
            self.rectangle_start_pos = event.pos()
            self.rectangle_start_x = int((self.rectangle_start_pos.x() - 305) * 640 / 960)
            self.rectangle_start_y = int((self.rectangle_start_pos.y() - 60) * 480 / 720)
        elif event.buttons() == QtCore.Qt.LeftButton and self.rectangle_start_pos != None and self.rectangle_end_pos == None and self.rectangleLabel==1:
            self.rectangle_end_pos = event.pos()
            self.rectangle_end_x = int((self.rectangle_end_pos.x() - 305) * 640 / 960)
            self.rectangle_end_y = int((self.rectangle_end_pos.y() - 60) * 480 / 720)
          
        elif event.buttons() == QtCore.Qt.LeftButton and self.circle_start_pos == None and self.circleLabel==1:
            self.circle_start_pos = event.pos()
            self.circle_start_x = int((self.circle_start_pos.x() - 305) * 640 / 960)
            self.circle_start_y = int((self.circle_start_pos.y() - 60) * 480 / 720)
        elif event.buttons() == QtCore.Qt.LeftButton and self.circle_start_pos != None and self.circle_end_pos == None and self.circleLabel==1:
            self.circle_end_pos = event.pos()
            self.circle_end_x = int((self.circle_end_pos.x() - 305) * 640 / 960)
            self.circle_end_y = int((self.circle_end_pos.y() - 60) * 480 / 720)


    def PrepCameraParameters(self):
        self.ExpTimeSld.setValue(self.camera.get(15))
        self.SetExposure()
        self.GainSld.setValue(self.camera.get(14))
        self.SetGain()
        self.BrightSld.setValue(self.camera.get(10))
        self.SetBrightness()
        self.ContrastSld.setValue(self.camera.get(11))
        self.SetContrast()
        self.MsgTE.clear()

    def CallBackFunctions(self):
        self.FilePathBt.clicked.connect(self.SetFilePath)
        self.ShowBt.clicked.connect(self.StartCamera)
        self.StopBt.clicked.connect(self.StopCamera)
        self.RecordBt.clicked.connect(self.RecordCamera)
        self.ExitBt.clicked.connect(self.ExitApp)
        # self.GrayImgCkB.stateChanged.connect(self.SetGray)
        self.ExpTimeSld.valueChanged.connect(self.SetExposure)
        self.GainSld.valueChanged.connect(self.SetGain)
        self.BrightSld.valueChanged.connect(self.SetBrightness)
        self.ContrastSld.valueChanged.connect(self.SetContrast)
        self.DetectBt.clicked.connect(self.ForegroundDetectionApp)
        self.KeypointsBt.clicked.connect(self.DetectKeypointsApp)
        self.CameraNOSld.valueChanged.connect(self.SetCameraNum)
        self.ThrSld.valueChanged.connect(self.SetAreaThreshold)
        self.GrayThrSld.valueChanged.connect(self.SetGrayThreshold)
        self.ChangeBgmBt.clicked.connect(self.ChangeBgmFlagApp)
        self.ScaleSpB.valueChanged.connect(self.SetScaleApp)
        self.SpeedThrSpB.valueChanged.connect(self.SetSpeedThrApp)
        self.resetRatio.clicked.connect(self.ResetRatio)
        self.rectangleBt.clicked.connect(self.RectangleArea)
        self.circleBt.clicked.connect(self.CircleArea)

    def DetectKeypointsApp(self):
        if self.KeypointsBt.text() == '姿态点检测':
            self.KeypointsBt.setText('停止姿态点')
            self.KeyPointsFlag = 1
        else:
            self.KeypointsBt.setText('姿态点检测')
            self.keyPoints = np.zeros([2, 8])
            self.keyPoints_all = []
            self.KeyPointsFlag = 0
            self.keyp_frame_num = 0

    def CircleArea(self):
        if self.circleBt.text() == '圆形选区':
            # self.rectangleLabel = 2
            self.circle_start_pos = None
            self.circle_end_pos = None
            self.circleBt.setText('取消选区')
            self.rectangleBt.setEnabled(False)
            self.circleLabel = 1
        else:
            self.circleBt.setText('圆形选区')
            self.circleLabel = 0
            self.rectangleBt.setEnabled(True)

    def RectangleArea(self):
        if self.rectangleBt.text() == '矩形选区':
            # self.rectangleLabel = 2
            self.rectangle_start_pos = None
            self.rectangle_end_pos = None
            self.rectangleBt.setText('取消选区')
            self.circleBt.setEnabled(False)
            self.rectangleLabel = 1
        else:
            self.rectangleBt.setText('矩形选区')
            self.rectangleLabel = 0
            self.circleBt.setEnabled(True)

    def ResetRatio(self):
        if self.resetRatio.text() == '重置比例':
            self.resetLabel = 2
            self.start_pos = None
            self.end_pos = None
            self.resetRatio.setText('擦除比例线')
            self.ratioLineLabel = 1
        else:
            self.resetRatio.setText('重置比例')
            self.ratioLineLabel = 0

    def PrepCamera(self):
        try:
            # print('cameraNO:', self.CameraNum)
            self.camera = cv2.VideoCapture(self.CameraNum, cv2.CAP_DSHOW)
            self.MsgTE.clear()
            self.MsgTE.append('Oboard camera connected.')
            self.MsgTE.setPlainText()
        except Exception as e:
            self.MsgTE.clear()
            self.MsgTE.append(str(e))

    def SetCameraNum(self):
        self.CameraNum = self.CameraNOSpB.value()
        self.PrepCamera()
        # print('cameraNO2:', self.CameraNum)

    def SetAreaThreshold(self):
        self.AreaThreshold = self.ThrSpB.value()

    def SetGrayThreshold(self):
        self.GrayThreshold = self.GrayThrSpB.value()

    def SetScaleApp(self):
        self.DisScale = self.ScaleSpB.value()

    def SetSpeedThrApp(self):
        self.SpeedThreshold = self.SpeedThrSpB.value()

    def SetContrast(self):
        contrast_toset=self.ContrastSld.value()
        try:
            self.camera.set(11,contrast_toset)
            self.MsgTE.setPlainText('The contrast is set to ' + str(self.camera.get(11)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))

    def SetBrightness(self):
        brightness_toset=self.BrightSld.value()
        try:
            self.camera.set(10,brightness_toset)
            self.MsgTE.setPlainText('The brightness is set to ' + str(self.camera.get(10)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))

    def SetGain(self):
        gain_toset=self.GainSld.value()
        try:
            self.camera.set(14, gain_toset)
            self.MsgTE.setPlainText('The gain is set to '+str(self.camera.get(14)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))

    def SetExposure(self):
        try:
            exposure_time_toset=self.ExpTimeSld.value()
            self.camera.set(15, exposure_time_toset)
            self.MsgTE.setPlainText('The exposure time is set to '+str(self.camera.get(15)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))

    def StartCamera(self):
        self.ShowBt.setEnabled(False)
        self.StopBt.setEnabled(True)
        self.RecordBt.setEnabled(True)
        self.DetectBt.setEnabled(True)
        self.KeypointsBt.setEnabled(True)
        self.ExpTimeSld.setEnabled(True)
        self.ExpTimeSpB.setEnabled(True)
        self.GainSld.setEnabled(True)
        self.GainSpB.setEnabled(True)
        self.BrightSld.setEnabled(True)
        self.BrightSpB.setEnabled(True)
        self.ContrastSld.setEnabled(True)
        self.ContrastSpB.setEnabled(True)
        self.RecordBt.setText('录像')
        self.Timer.start(1)
        # self.timelb=time.clock()
        self.timelb=time.perf_counter()

        # # 打开摄像头的时候初始化网络模型
        background_name = self.RecordPath + 'Background.jpg'
        self.Background = cv2.imread(background_name, cv2.IMREAD_COLOR)
        # # 网络预测
        time_start = time.time()
        self.MsgTE.setPlainText('Loading the KeyPoints Net...')
        self.net, self.device = initNet()  # , center)
        print(000)
        heatmap = predict_img(net=self.net, full_img=self.Background, device=self.device, resize_w=320, resize_h=256)
        time_end = time.time()
        print('net-time:', time_end-time_start)

    def SetFilePath(self):
        dirname = QFileDialog.getExistingDirectory(self, "浏览", '.')
        if dirname:
            self.FilePathLE.setText(dirname)
            self.RecordPath=dirname+'/'

    def TimerOutFun(self):
        frame_num = 0
        if self.DetectFlag:
            # background_name = self.RecordPath + 'Background.jpg'
            # if frame_num==0:
            #     self.MsgTE.setPlainText('Loading the Background...')
            # else:
            self.MsgTE.setPlainText('Detecting...')
            # self.Background = cv2.imread(background_name, cv2.IMREAD_COLOR)


            success, img = self.camera.read()

            if success:
                self.Image = img  # self.ColorAdjust(img)
                self.original_img = img   # 保存时同时保存原视频
                if self.RecordFlag:
                    self.video_writer_original.write(self.original_img)

                ###########################################姿态点检测##########################################
                if self.KeyPointsFlag == 1:
                    # time_start = time.time()
                    # if self.keyp_frame_num % 2 == 0:
                    #     self.keyPoints, self.Angle, self.keyPoints_all = PoseDetection(self.Image, 640, 480,
                    #                                                               self.device, self.net, 10,
                    #                                                               self.keyPoints, self.keyPoints_all)
                    #
                    #     # print('deted')
                    # else:
                    #     self.keyPoints, self.keyPoints_all = noDetection(self.keyp_frame_num, 8, self.keyPoints,
                    #                                                      self.keyPoints_all)
                    self.keyPoints, self.Angle, self.keyPoints_all = PoseDetection(self.Image, 640, 480,
                                                                                   self.device, self.net, 10,
                                                                                   self.keyPoints, self.keyPoints_all)
                        # print('nodeted')
                    # time_end = time.time()
                    if len(self.keyPoints_all) > 4:
                        self.keyPoints_all = self.keyPoints_all[-4:]
                    self.keyp_frame_num = self.keyp_frame_num + 1
                    # print('net--time:', time_end - time_start)

                    center = [np.mean(self.keyPoints[0, :]), np.mean(self.keyPoints[1, :])]   # 根据关键点计算中心点
                    self.pts.append(center)
                ###########################################姿态点检测##########################################


                # 背景建模
                self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 形态学操作需要使用
                self.fgbg = cv2.createBackgroundSubtractorMOG2()  # 创建混合高斯模型用于背景建模
                ###########################################鼠标划分区域##########################################
                #  圆形选区
                if self.circleLabel==1:
                    self.Image, radius, mid_x, mid_y = DrawcircleArea(self.Image, self.circle_start_pos,
                                                self.circle_end_pos, self.circle_start_x, self.circle_start_y,
                                                self.circle_end_x, self.circle_end_y)
                    # print('r:', radius)

                # 矩形选区
                if self.rectangleLabel==1:
                    self.Image = DrawRectangleArea(self.Image, self.rectangle_start_pos, self.rectangle_end_pos,
                                                 self.rectangle_start_x, self.rectangle_start_y, self.rectangle_end_x,
                                                 self.rectangle_end_y)

                if self.circleLabel==1 and self.circle_end_pos != None:
                        # print('x', mid_x)
                        Background_circle = CropCircleArea2(self.Background, mid_x, mid_y, radius)
                        # print('y', mid_y)
                        Foreground, rect = CalculateForegrounds(Background_circle, self.Image, self.kernel, self.pts,
                                                                self.ChangeBgmFlag, self.AreaThreshold,
                                                                self.GrayThreshold, self.signalLamp, self.KeyPointsFlag)
                        if rect != 0:
                            centerPoints = np.int0(rect[0])
                elif self.rectangleLabel==1 and self.rectangle_end_pos != None:
                    # img = self.Image - self.Image
                    # rect=[]
                    img0 = self.Image[
                           self.rectangle_start_y:self.rectangle_end_y, self.rectangle_start_x:self.rectangle_end_x]
                    # bg=self.Background-self.Background
                    bg = self.Background[self.rectangle_start_y:self.rectangle_end_y,
                                                                   self.rectangle_start_x:self.rectangle_end_x]
                    foreground, rect = CalculateForegrounds(bg, img0, self.kernel, self.pts,
                                                            self.ChangeBgmFlag, self.AreaThreshold,
                                                            self.GrayThreshold, self.signalLamp, self.KeyPointsFlag)

                    if rect != 0:
                        centerPoints = np.int0(rect[0])
                        centerPoints[0] = centerPoints[0] + self.rectangle_start_x
                        centerPoints[1] = centerPoints[1] + self.rectangle_start_y

                    Foreground = self.Image
                    Foreground[self.rectangle_start_y:self.rectangle_end_y,
                               self.rectangle_start_x:self.rectangle_end_x]= foreground

                else:
                    Foreground, rect = CalculateForegrounds(self.Background, self.Image, self.kernel, self.pts,
                                                            self.ChangeBgmFlag, self.AreaThreshold,
                                                            self.GrayThreshold, self.signalLamp, self.KeyPointsFlag)
                    if rect != 0:
                        centerPoints = np.int0(rect[0])
                        # print('center:', centerPoints)
                ###########################################鼠标划分区域##########################################

                if rect==0:
                    self.DispImg()
                    self.MsgTE.setPlainText('No Object Detected.')
                else:
                    #########################方向判断###########################
                    # center=(rect[0][0], rect[0][1])
                    if self.KeyPointsFlag == 0:   # 如果未检测关键点，则将背景建模的中心点写入中心点序列，否则中心点默认为关键点均值
                        self.pts.append(centerPoints)
                    # if len(self.pts) > 7:
                    #     s = JudgeDeriection(np.array(self.pts))
                    #     if s > 0:
                    #         print('左转')
                    #     elif s < 0:
                    #         print('右转')
                    #     else:
                    #         print('直线')
                    # print(len(self.pts))
                    # cv2.imshow("rstp", Foreground)
                    #########################方向判断###########################
                    self.Image = Foreground
                    ########################视频上绘制时间##############################
                    TimeText = 'Time:' + datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S.%f')
                    cv2.putText(self.Image, TimeText, (370, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                                      cv2.LINE_AA)
                    ########################视频上绘制时间##############################

                    ########################视频上绘制信号灯和中心点坐标##############################
                    if self.signalLamp == 2:  # 信号灯
                        cv2.circle(self.Image, (20, 30), 10, (0, 0, 255), -1)  # 速度大于100时亮红灯
                    elif self.signalLamp == 1:  # 信号灯
                        cv2.circle(self.Image, (20, 30), 10, (0, 255, 0), -1)  # 速度在[50, 100)区间时亮绿灯
                    # 添加中心点坐标 centerPoints[0], centerPoints[1]
                    CenterText = 'Center: [' + str(centerPoints[0]) + ' ' + str(centerPoints[1]) + ']'
                    cv2.putText(self.Image, CenterText, (480, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                                      cv2.LINE_AA)
                    ########################视频上绘制信号灯和中心点坐标##############################

                    ################################绘制关键点#####################################
                    if self.KeyPointsFlag == 1:
                        for j in range(8):
                            cv2.circle(self.Image, (int(self.keyPoints[0, j]), int(self.keyPoints[1, j])), 2,
                                       (0, 0, 255), 4)

                        self.Image = draw_relation(self.Image, self.keyPoints, self.relation)
                        cv2.arrowedLine(self.Image, (int(self.keyPoints[0, 7] + 30), int(self.keyPoints[1, 7] + 30)),
                                        (int(self.keyPoints[0, 6] + 30), int(self.keyPoints[1, 6]) + 30), (0, 0, 255),
                                        thickness=2, line_type=cv2.LINE_4, shift=0, tipLength=0.5)
                        AngleText = 'Angle: [' + str(self.Angle) + ']'
                        cv2.putText(self.Image, AngleText, (420, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                                    cv2.LINE_AA)
                        # print('angle:', self.Angle)
                        self.AngleLCD.display(self.Angle)
                    else:
                        self.AngleLCD.display(0)
                    ################################绘制关键点#####################################


                    ###显示身长
                    length = np.int0(np.max(rect[1]))
                    length = length / self.DisScale  #一厘米约等于2.44个像素
                    self.LengthLCD.display(length)
                    # angle = np.int0(rect[2])
                    # 实时速度计算
                    # time2 = time.time()
                    # cx2 = centerPoints[0]
                    # cy2 = centerPoints[1]
                    # if self.frame_num == 0:
                    #     self.cx1 = cx2
                    #     self.cy1 = cy2
                    #     self.time1 = time2
                    # # else:
                    # if self.frame_num<=5:
                    #     self.speed=0
                    # elif self.frame_num % 5 == 0:  # 每5帧计算一次速度
                    #     self.speed = CalculateSpeed(self.cx1, self.cy1, cx2, cy2, self.time1, time2, self.DisScale)
                    #     self.time1 = time2
                    #     self.cx1 = cx2
                    #     self.cy1 = cy2
                    #     self.SpeedAll.append(self.speed)
                    self.SpeedAll = get_speed(self.pts, self.DisScale, self.frame_rate, window_len=10)   # 速度计算
                    if len(self.pts) <= 10:
                        self.SpeedAll = [0]
                    self.speed = self.SpeedAll[-1]

                    SpeedText = '  Speed: [' + str(round(self.speed, 2)) + ']'
                    cv2.putText(self.Image, SpeedText, (510, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                                cv2.LINE_AA)

                    self.DispImg()

                    ### 根据最近几帧的速度变换判断是否亮指示灯（接近静止状态时亮红灯）
                    # if len(self.SpeedAll) > 3:
                    #     if max(self.SpeedAll[len(self.SpeedAll)-4:len(self.SpeedAll)-1]) <= self.SpeedThreshold:
                    #         self.signalLamp = 1
                    #     else:
                    #         self.signalLamp = 0

                    if self.speed >= 30 and self.speed < 100:  # 速度在[50, 100)区间时亮绿灯
                        self.signalLamp = 1
                    elif self.speed >= 100:                    # 速度大于100时亮红灯
                        self.signalLamp = 2
                    else:
                        self.signalLamp = 0

                    self.SpeedLCD.display(self.speed)

                # if self.frame_num%10 ==0:

            self.frame_num = self.frame_num + 1
            # 保存\显示

            if self.RecordFlag:
                # self.video_writer_original.write(self.original_img)
                self.video_writer.write(Foreground)

                #########################################写记录文档csv############################################
                with open(self.LogCSVPath, "a", newline='') as datacsv:
                    # dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
                    csvwriter = csv.writer(datacsv, dialect=("excel"))
                    # csv文件插入一行数据，把下面列表中的每一项放入一个单元格（可以用循环插入多行）
                    if self.KeyPointsFlag == 1:
                        csvwriter.writerow([centerPoints[0], centerPoints[1], round(length, 2),
                                            round(self.speed, 2), self.Angle,
                                            self.keyPoints[0][0], self.keyPoints[1][0],
                                            self.keyPoints[0][1], self.keyPoints[1][1],
                                            self.keyPoints[0][2], self.keyPoints[1][2],
                                            self.keyPoints[0][3], self.keyPoints[1][3],
                                            self.keyPoints[0][4], self.keyPoints[1][4],
                                            self.keyPoints[0][5], self.keyPoints[1][5],
                                            self.keyPoints[0][6], self.keyPoints[1][6],
                                            self.keyPoints[0][7], self.keyPoints[1][7],
                                            self.signalLamp, self.AreaThreshold,
                                            datetime.datetime.now().strftime('%Y%m%d'),
                                            datetime.datetime.now().strftime('%H:%M:%S:%f')])
                    elif self.DetectFlag == 1:
                        csvwriter.writerow([centerPoints[0], centerPoints[1], round(length, 2),
                                            round(self.speed, 2), 0,
                                            0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0,
                                            self.signalLamp, self.AreaThreshold,
                                            datetime.datetime.now().strftime('%Y%m%d'),
                                            datetime.datetime.now().strftime('%H:%M:%S:%f')])
                    else:
                        csvwriter.writerow([0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0,
                                            datetime.datetime.now().strftime('%Y%m%d'),
                                            datetime.datetime.now().strftime('%H:%M:%S:%f')])

                #########################################写记录文档csv############################################

            if self.frame_num % 10 == 9:
                # frame_rate=10/(time.clock()-self.timelb)
                self.frame_rate = 10 / (time.perf_counter() - self.timelb)
                self.FmRateLCD.display(round(self.frame_rate, 1))
                # self.timelb=time.clock()
                self.timelb = time.perf_counter()
                # size=img.shape
                self.ImgWidthLCD.display(self.camera.get(3))
                self.ImgHeightLCD.display(self.camera.get(4))
        else:
            success, img = self.camera.read()
            if success:
                # self.Image = self.ColorAdjust(img)
                self.Image = img

                ###################################################比例尺######################################################
                if self.start_pos != None and self.end_pos != None and self.ratioLineLabel==1:
                    if self.RatioLengthSpB.value() != 0:
                        self.DisScale = round(np.sqrt((self.start_x - self.end_x) ** 2 + (
                                self.start_y - self.end_y) ** 2) / self.RatioLengthSpB.value(), 2)

                    if self.ScaleSpB.value != self.DisScale:
                        self.ScaleSpB.setValue(self.DisScale)
                    cv2.line(self.Image, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 0, 155), 1)
                    # cv2.line(self.Image, (int(424 - 424) * 480 / 651), int((65 - 65) * 640 / 841), (int((1265 - 424) * 480 / 651), int((716 - 65) * 640 / 841)), (255, 0, 155), 1)
                    # print(1)
                if self.start_pos != None and self.ratioLineLabel == 1:
                    cv2.circle(self.Image, (self.start_x, self.start_y), 2, (255, 0, 155), 1)
                if self.end_pos != None and self.ratioLineLabel == 1:
                    cv2.circle(self.Image, (self.end_x, self.end_y), 2, (255, 0, 155), 1)
                ###################################################比例尺######################################################

                # 矩形选区
                if self.rectangleLabel==1:
                    self.Image=DrawRectangleArea(self.Image, self.rectangle_start_pos, self.rectangle_end_pos,
                                                 self.rectangle_start_x, self.rectangle_start_y, self.rectangle_end_x,
                                                 self.rectangle_end_y)
                #  圆形选区
                if self.circleLabel==1:
                    self.Image, radius, mid_x, mid_y = DrawcircleArea(self.Image, self.circle_start_pos,
                                                self.circle_end_pos, self.circle_start_x, self.circle_start_y,
                                                self.circle_end_x, self.circle_end_y)

                self.DispImg()
                self.Image_num += 1
                if self.RecordFlag:
                    self.video_writer.write(img)
                    #########################################写记录文档csv############################################
                    with open(self.LogCSVPath, "a", newline='') as datacsv:
                        # dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
                        csvwriter = csv.writer(datacsv, dialect=("excel"))
                        # csv文件插入一行数据，把下面列表中的每一项放入一个单元格（可以用循环插入多行）

                        csvwriter.writerow([0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0,
                                            datetime.datetime.now().strftime('%Y%m%d'),
                                            datetime.datetime.now().strftime('%H:%M:%S:%f')])

                    #########################################写记录文档csv############################################


                if self.Image_num % 10 == 9:
                    # frame_rate=10/(time.clock()-self.timelb)
                    self.frame_rate = 10 / (time.perf_counter() - self.timelb)
                    self.FmRateLCD.display(self.frame_rate)
                    # self.timelb=time.clock()
                    self.timelb = time.perf_counter()
                    # size=img.shape
                    self.ImgWidthLCD.display(self.camera.get(3))
                    self.ImgHeightLCD.display(self.camera.get(4))
            else:
                self.MsgTE.clear()
                self.MsgTE.setPlainText('Image obtaining failed.')

    def DispImg(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
        qimg = qimage2ndarray.array2qimage(img)
        self.DispLb.setPixmap(QPixmap(qimg))
        self.DispLb.show()

    def StopCamera(self):
        if self.StopBt.text() == '暂停':
            self.StopBt.setText('继续')
            self.RecordBt.setText('保存')
            self.Timer.stop()
        elif self.StopBt.text() == '继续':
            self.StopBt.setText('暂停')
            self.RecordBt.setText('录像')
            self.Timer.start(1)

    def RecordCamera(self):
        tag = self.RecordBt.text()
        if tag == '保存':
            try:
                # image_name=self.RecordPath+'image'+time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))+'.jpg'
                image_name = self.RecordPath + 'Background.jpg'
                if not os.path.exists(self.RecordPath):
                    os.makedirs(self.RecordPath)
                # print(image_name)
                cv2.imwrite(image_name, self.Image)
                self.MsgTE.clear()
                self.MsgTE.setPlainText('Image saved.')
                self.Background = self.Image
            except Exception as e:
                self.MsgTE.clear()
                self.MsgTE.setPlainText(str(e))
        elif tag=='录像':
            self.RecordBt.setText('停止')
            video_name = self.RecordPath + 'video' + time.strftime('%Y%m%d%H%M%S',time.localtime(time.time())) + '.avi'
            video_original = self.RecordPath + 'original_video' + time.strftime('%Y%m%d%H%M%S',
                                                                                time.localtime(time.time())) + '.avi'
            fps = self.FmRateLCD.value()
            size = (self.Image.shape[1],self.Image.shape[0])
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            self.video_writer = cv2.VideoWriter(video_name, fourcc, fps, size)  # self.camera.get(5), size)
            self.video_writer_original = cv2.VideoWriter(video_original, fourcc, fps, size)  # self.camera.get(5), size)
            self.RecordFlag=1
            self.MsgTE.setPlainText('Video recording...')
            self.StopBt.setEnabled(False)
            self.ExitBt.setEnabled(False)

            # 初始化记录文档
            self.LogPath = self.RecordPath + 'Logs\\'
            if not os.path.exists(self.LogPath):
                os.makedirs(self.LogPath)
            # self.LogPath=self.LogPath + 'Logs_' + time.strftime('%Y%m%d%H%M%S',time.localtime(time.time())) + '.txt'
            self.LogCSVPath = self.LogPath + 'Logs_' + time.strftime('%Y%m%d%H%M%S',
                                                                     time.localtime(time.time())) + '.csv'
            # 写记录文档表头
            with open(self.LogCSVPath, "a", newline='') as datacsv:
                # dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
                csvwriter = csv.writer(datacsv, dialect=("excel"))
                # csv文件插入一行数据，把下面列表中的每一项放入一个单元格（可以用循环插入多行）
                csvwriter.writerow(["中心坐标x", "中心坐标y", "身长", "速度", "角度",
                                    'rRP_x', 'rRP_y',
                                    'lRP_x', 'lRP_y',
                                    'rFP_x', 'rFP_y',
                                    'lFP_x', 'lFP_y',
                                    'tail_x', 'tail_y',
                                    'head_x', 'head_y',
                                    'neck_x', 'neck_y',
                                    'spine_x', 'spine_y',
                                    "信号灯状态", "检测阈值", "日期(年月日)", "时间(时：分：秒：毫秒)"])

        elif tag == '停止':
            self.RecordBt.setText('录像')
            self.video_writer.release()
            self.video_writer_original.release()
            self.RecordFlag = 0
            self.MsgTE.setPlainText('Video saved.')
            self.StopBt.setEnabled(True)
            self.ExitBt.setEnabled(True)

    def ExitApp(self):
        # self.Timer.Stop()
        self.camera.release()
        self.MsgTE.setPlainText('Exiting the application..')
        QCoreApplication.quit()

    #########################################
    def DispForeground(self):
        Foreground = CalculateForegrounds(self.Background, self.Image)
        qimg = qimage2ndarray.array2qimage(Foreground)
        self.DispLb.setPixmap(QPixmap(qimg))
        self.DispLb.show()

    def ForegroundDetectionApp(self):
        if self.DetectBt.text() == '检测':
            self.DetectBt.setText('停止检测')
            # self.StopBt.setEnabled(False)
            # self.RecordBt.setEnabled(False)
            self.DetectFlag = 1


        elif self.DetectBt.text() == '停止检测':
            self.DetectBt.setText('检测')
            self.StopBt.setEnabled(True)
            self.RecordBt.setEnabled(True)
            self.DetectFlag = 0
            self.pts=[]
            # 停止检测时，关键点也清零
            self.keyPoints = np.zeros([2, 8])
            self.keyPoints_all = []
            self.KeypointsBt.setText('姿态点检测')
            self.KeyPointsFlag = 0
            self.keyp_frame_num = 0

    def ChangeBgmFlagApp(self):
        if self.ChangeBgmBt.text() == '切换至浅色背景':
            self.ChangeBgmBt.setText('切换至深色背景')
            self.ChangeBgmFlag = 1
        elif self.ChangeBgmBt.text() == '切换至深色背景':
            self.ChangeBgmBt.setText('切换至浅色背景')
            self.ChangeBgmFlag = 0

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet())
    ui=CamShow()
    ui.show()
    sys.exit(app.exec_())
