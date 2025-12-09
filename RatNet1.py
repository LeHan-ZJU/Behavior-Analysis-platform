import numpy as np
import os
import csv
import cv2
import time
import math
import torch
import logging
import torch.nn.functional as F
from torchvision import models
# from RatNet.Models import FeatureExtractor, RatNet_Resnet2
from RatNet.RatNetAttention_DOConv import Net_ResnetAttention_DOConv


def preprocess(resize_w, resize_h, pil_img):
    pil_img = cv2.resize(pil_img, (resize_w, resize_h))
    img_nd = np.array(pil_img)
    if len(img_nd.shape) == 2:
        img_nd0 = img_nd
        img_nd = np.expand_dims(img_nd0, axis=2)
        img_nd = np.concatenate([img_nd, img_nd, img_nd], axis=-1)
    img_nd = img_nd.transpose((2, 0, 1))
    if img_nd.max() > 1:
        img_nd = img_nd / 255
    return img_nd


def predict_img(net,
                full_img,
                device,
                resize_w,
                resize_h,
                ):
    net.eval()

    img = torch.from_numpy(preprocess(resize_w, resize_h, full_img))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        probs = F.softmax(output, dim=1)

        probs = probs.squeeze(0)
        output = probs.cpu()
    return output


def heatmap_to_image(img, heatmap, i):
    NormMap = (heatmap[i, :, :] - np.mean(heatmap[i, :, :])) / (np.max(heatmap[i, :, :]) - np.mean(heatmap[i, :, :]))
    map = np.round(NormMap * 255)
    map = cv2.resize(map, (640, 480))
    img = img*0.3
    img[:, :, 1] = map + img[:, :, 1]
    return img


def heatmap_to_points_0(i, Img, heatmap, numPoints, ori_W, ori_H, keyPoints, keyP_all):
    keyP_all = np.array(keyP_all)
    # print(keyP_all, keyP_all.shape)
    for j in range(numPoints):
        hm = cv2.resize(heatmap[j, :, :], (ori_W, ori_H))
        center = np.unravel_index(np.argmax(hm), hm.shape)
        cx = center[1]
        cy = center[0]
        if i > 2:
            dis = np.sqrt((cx-keyP_all[i-2, 0, j])**2 + (cy-keyP_all[i-2, 1, j])**2)
            if dis < 180:
                keyPoints[0, j] = cx
                keyPoints[1, j] = cy
            else:
                delt = keyP_all[i - 2, :, :] - keyP_all[i - 3, :, :]
                delt = np.mean(delt, axis=1)
                keyPoints[:, j] = keyP_all[i - 1, :, j] + delt
        else:
            keyPoints[0, j] = cx
            keyPoints[1, j] = cy
        # keyPoints[0, j] = cx
        # keyPoints[1, j] = cy
        cv2.circle(Img, (int(keyPoints[0, j]), int(keyPoints[1, j])), 2, (0, 0, 255), 4)
    return Img, keyPoints


def heatmap_to_points(heatmap, numPoints, ori_W, ori_H, keyPoints, keyP_all):
    keyP_all = np.array(keyP_all)
    # print(keyP_all, keyP_all.shape)
    for j in range(numPoints):
        hm = cv2.resize(heatmap[j, :, :], (ori_W, ori_H))
        center = np.unravel_index(np.argmax(hm), hm.shape)
        cx = center[1]
        cy = center[0]
        if len(keyP_all) == 4:
            dis = np.sqrt((cx-keyP_all[2, 0, j])**2 + (cy-keyP_all[2, 1, j])**2)
            if dis < 180:
                keyPoints[0, j] = cx
                keyPoints[1, j] = cy
            else:
                delt = keyP_all[2, :, :] - keyP_all[0, :, :]
                delt = np.mean(delt, axis=1)/2
                keyPoints[:, j] = keyP_all[2, :, j] + delt
        else:
            keyPoints[0, j] = cx
            keyPoints[1, j] = cy
        # keyPoints[0, j] = cx
        # keyPoints[1, j] = cy
        # cv2.circle(Img, (int(keyPoints[0, j]), int(keyPoints[1, j])), 2, (0, 0, 255), 4)
    if len(keyP_all) > 4:
        keyPoints = RefineKeyPoints(keyPoints, keyP_all, numPoints)

    return keyPoints


def draw_relation(Img, allPoints, relations):
    for k in range(len(relations)):
        c_x1 = int(allPoints[0, relations[k][0]-1])
        c_y1 = int(allPoints[1, relations[k][0]-1])
        c_x2 = int(allPoints[0, relations[k][1]-1])
        c_y2 = int(allPoints[1, relations[k][1]-1])
        cv2.line(Img, (c_x1, c_y1), (c_x2, c_y2), (0, 255, 0), 2)
    return Img


def CalAngle(points):
    p1 = points[:, 6]
    p2 = points[:, 7]
    delt_x = p1[0] - p2[0]
    delt_y = p1[1] - p2[1]

    if delt_x == 0:
        if delt_y <= 0:
            A = 90
        else:
            A = 270
    else:
        A = math.degrees(math.atan(np.abs((p1[1] - p2[1]) / (p1[0] - p2[0]))))
        if delt_x>=0 and delt_y >= 0:   # D
            A = 360 - A
        if delt_x < 0 and delt_y >= 0:   # C
            A = 180 + np.abs(A)
        elif delt_x < 0 and delt_y < 0:   # B
            A = 180 - A
        elif delt_x >= 0 and delt_y < 0:   # A
            A = np.abs(A)
    return int(A), delt_x, delt_x


def get_dis(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_speed(center_list, window_len=10):
    speed_list = []
    for i in range(len(center_list) - 10):
        cur_dis = 0
        for j in range(window_len):
            cur_dis += get_dis(center_list[i+j], center_list[i+j+1])

        speed_list.append((cur_dis * 24) / 34)  # 10frame*3.4pix/s
    return speed_list


# def draw_speed(speeds):
#     x = np.linspace(21, 620, 600)
#     y = np.ones(1, 600)


def RefineKeyPoints(keyps, keyps_all, num_points):
    keyps_all = np.array(keyps_all)
    keyps_pre = keyps_all[-1, :, :]
    for p in range(num_points):
        points = keyps
        points[:, p] = [0, 0]
        cent = [sum(points[0, :])/(num_points - 1), sum(points[1, :])/(num_points - 1)]
        if np.sqrt((cent[0]-keyps[0, p])**2 + (cent[1]-keyps[1, p])**2) > 120:
            points_pre = keyps_pre
            points_pre[:, p] = [0, 0]
            cent_pre = [sum(points_pre[0, :]) / (num_points - 1), sum(points_pre[1, :]) / (num_points - 1)]
            # delt = keyP_all[2, :, :] - keyP_all[0, :, :]
            delt = keyps_pre[:, p] - cent_pre
            keyps[:, p] = cent + delt
    return keyps



# 构建网络
def RatNet(device):
    extract_list = ["layer4"]

    net = Net_ResnetAttention_DOConv('none', extract_list, device, train=False, n_channels=3,
                                     nof_joints=10)

    print('Using device ', device)
    net.to(device=device)
    net.load_state_dict(torch.load('./RatNet/models.pth', map_location=device), strict=False)
    print("Model loaded !")
    return net  # heatmaps


def PoseDetection(img, ori_w, ori_h, device, net, num_points, keyPoints, keyPoints_all):
    heatmap = predict_img(net=net,
                          full_img=img,
                          device=device,
                          resize_w=320,
                          resize_h=256)
    heatmap = heatmap.numpy().reshape((num_points, 256 // 4, 320 // 4))
    keyPoints = heatmap_to_points(heatmap, num_points - 2, ori_w, ori_h, keyPoints, keyPoints_all)
    # keyPoints = AnomalyDetection(i, keyPoints, keyPoints_all, num_points-2, 70)
    # print(3, keyPoints)
    angle, deltX, deltY = CalAngle(keyPoints)
    # img = draw_relation(img, keyPoints, relation)
    # cv2.arrowedLine(img, (int(keyPoints[0, 7] + 30), int(keyPoints[1, 7] + 30)),
    #                 (int(keyPoints[0, 6] + 30), int(keyPoints[1, 6]) + 30), (0, 0, 255), thickness=2,
    #                 line_type=cv2.LINE_4, shift=0, tipLength=0.5)
    keyPoints_all.append(keyPoints)
    # keyPoints_all = np.array(keyPoints_all)
    return keyPoints, angle, keyPoints_all


def noDetection(i, num_points, keyPs, keyP_all):
    # print('i:', i)
    keyP_all_arr = np.array(keyP_all)
    # print('keyP_all_arr:', keyP_all_arr.shape)
    if i == 1:
        keyPs = keyP_all_arr[-1, :, :]
    else:  # 计算delt
        delt = keyP_all_arr[1, :, :] - keyP_all_arr[0, :, :]

        for n in range(num_points):
            keyPs[:, n] = keyP_all_arr[-1, :, n] + delt[:, n]

    keyP_all.append(keyPs)
    return keyPs, keyP_all


def initNet():  # , center):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    net = RatNet(device)
    return net, device  # img, keyPoints


if __name__ == "__main__":
    in_files = 'G:/Data/LHY_Data/2021.12.10/origin_1-1-20211210153807.avi'
    out_files = './RatNet/debugResults/'

    isExists = os.path.exists(out_files)
    if not isExists:
        os.makedirs(out_files)
    excel_path = out_files + 'result.csv'
    saved_path = out_files + 'result2.avi'
    # relation = [[1, 8], [2, 8], [3, 7], [4, 7], [6, 7], [7, 8], [5, 8]]
    relation = [[1, 5], [1, 8], [1, 7], [2, 5], [2, 7], [2, 8], [3, 7], [3, 6],
                [4, 6], [4, 7], [5, 8], [7, 8]]
    num_points = 10
    resize_w = 320
    resize_h = 256

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = RatNet(device)

    with open(excel_path, "a", newline='') as datacsv:
        csvwriter = csv.writer(datacsv, dialect="excel")
        csvwriter.writerow(['frame_num', 'angle', 'speed',
                            'rRP_x', 'rRP_y',
                            'lRP_x', 'lRP_y',
                            'rFP_x', 'rFP_y',
                            'lFP_x', 'lFP_y',
                            'tail_x', 'tail_y',
                            'head_x', 'head_y',
                            'neck_x', 'neck_y',
                            'spine_x',  'spine_y'])

    cap = cv2.VideoCapture(in_files)
    ori_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ori_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('视频帧率:', fps, ' 总帧数：', num_frames, ' 每一帧图像尺寸：', ori_height, ori_width)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_writer = cv2.VideoWriter(saved_path, fourcc, 30, (ori_width, ori_height))

    i = 0
    keyPoints = np.zeros([2, num_points-2])
    # keyPoints_all = np.zeros([4, 2, num_points-2])
    keyPoints_all = []
    center_all = []
    while i < num_frames:
        ret, frame0 = cap.read()
        # print('frame0:', frame0.shape)
        # img = cv2.resize(frame0, (ori_width, ori_height))
        # print('img:', img.shape)

        time_start = time.time()
        # heatmap = predict_img(net=net,
        #                       full_img=frame0,
        #                       device=device,
        #                       resize_w=resize_w,
        #                       resize_h=resize_h)
        #
        # heatmap = heatmap.numpy().reshape((num_points, resize_h//4, resize_w//4))
        # frame0, keyPoints = heatmap_to_points(i, frame0, heatmap, num_points-2, ori_width, ori_height, keyPoints, keyPoints_all)
        # # keyPoints = AnomalyDetection(i, keyPoints, keyPoints_all, num_points-2, 70)
        # frame0 = draw_relation(frame0, keyPoints, relation)
        #
        # angle, deltX, deltY = CalAngle(keyPoints)

        # keyPoints, angle, keyPoints_all = PoseDetection(frame0, ori_width, ori_height, device, net,
        #                                                 num_points, keyPoints, keyPoints_all)
        if i % 2 == 0:
            keyPoints, angle, keyPoints_all = PoseDetection(frame0, ori_width, ori_height, device, net,
                                                                    num_points, keyPoints, keyPoints_all)
            # print('angle1:', angle)
        else:
            keyPoints, keyPoints_all = noDetection(i, num_points-2, keyPoints, keyPoints_all)
        for j in range(num_points-2):
            cv2.circle(frame0, (int(keyPoints[0, j]), int(keyPoints[1, j])), 2, (0, 0, 255), 4)
                # print('angle2:', angle)
        if len(keyPoints_all) > 4:
            keyPoints_all = keyPoints_all[-4:]
            # print(keyPoints_all)

        img = draw_relation(frame0, keyPoints, relation)
        cv2.arrowedLine(frame0, (int(keyPoints[0, 7] + 30), int(keyPoints[1, 7] + 30)),
                        (int(keyPoints[0, 6] + 30), int(keyPoints[1, 6]) + 30), (0, 0, 255), thickness=2,
                        line_type=cv2.LINE_4, shift=0, tipLength=0.5)

        time_end = time.time()
        if i % 100 == 0:
            print('time:', time_end - time_start, i)

        center = [np.mean(keyPoints[0, :]), np.mean(keyPoints[1, :])]
        # print(keyPoints, center)
        center_all.append(center)
        # 计算速度
        speeds = get_speed(center_all, 10)
        if len(center_all) <= 10:
            speeds = [0]

        frameText = 'frame: [' + str(i) + ']'
        cv2.putText(frame0, frameText, (520, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        angleText = 'angle: [' + str(angle) + ']'
        cv2.putText(frame0, str(angleText), (520, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        speedText = 'speed: [' + str(round(speeds[-1], 2)) + ']'
        cv2.putText(frame0, str(speedText), (520, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        video_writer.write(frame0)

        # keyPoints_all[i, :, :] = keyPoints
        # keyPoints_all.append(keyPoints)

        with open(excel_path, "a", newline='') as datacsv:
            csvwriter = csv.writer(datacsv, dialect="excel")
            csvwriter.writerow([i, angle, speeds[-1],
                                keyPoints[0][0], keyPoints[1][0],
                                keyPoints[0][1], keyPoints[1][1],
                                keyPoints[0][2], keyPoints[1][2],
                                keyPoints[0][3], keyPoints[1][3],
                                keyPoints[0][4], keyPoints[1][4],
                                keyPoints[0][5], keyPoints[1][5],
                                keyPoints[0][6], keyPoints[1][6],
                                keyPoints[0][7], keyPoints[1][7]])

        i = i + 1