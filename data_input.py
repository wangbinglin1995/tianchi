import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt

import scipy.io as sio
import cv2
import os
import random
import PIL


def get_train_data(re_size, is_std):
    path_N = r'../guangdong_round1_train2_20180916/无瑕疵样本'
    path_1 = r'../guangdong_round1_train2_20180916/瑕疵样本/不导电'
    path_2 = r'../guangdong_round1_train2_20180916/瑕疵样本/擦花'
    path_3 = r'../guangdong_round1_train2_20180916/瑕疵样本/横条压凹'
    path_4 = r'../guangdong_round1_train2_20180916/瑕疵样本/桔皮'
    path_5 = r'../guangdong_round1_train2_20180916/瑕疵样本/漏底'
    path_6 = r'../guangdong_round1_train2_20180916/瑕疵样本/碰伤'
    path_7 = r'../guangdong_round1_train2_20180916/瑕疵样本/起坑'
    path_8 = r'../guangdong_round1_train2_20180916/瑕疵样本/凸粉'
    path_9 = r'../guangdong_round1_train2_20180916/瑕疵样本/涂层开裂'
    path_10 = r'../guangdong_round1_train2_20180916/瑕疵样本/脏点'
    path_11 = r'../guangdong_round1_train2_20180916/瑕疵样本/其他2'

        
    paths = [path_N, path_1, path_2, path_3, path_4, path_5,
                 path_6, path_7, path_8, path_9, path_10, path_11]
        
    Data = []
    Labels = []
    for i in range(len(paths)):
        for file in os.listdir(paths[i]):
            file_path_raw = os.path.join(paths[i], file)
                
               
            # torch :RGB,CHW
            # PIL: RGB, HWC
            # cv2: BGR, HWC
            img_obj = PIL.Image.open(file_path_raw)
            img_obj = img_obj.resize(re_size, PIL.Image.ANTIALIAS)

            img = np.array(img_obj, dtype='float32')
                
            if is_std:
                # img[:, :, 1] = img[:, :, 0]
                # img[:, :, 2] = img[:, :, 0]
                img = img / 255
                # img = img - 0.5

            Data.append(img)
            Labels.append(i)
            
    Data = np.array(Data)
    Labels = np.array(Labels)
    return Data, Labels
                
               
        
def get_test_data(re_size, is_std):
    path = r'../guangdong_round1_test_a_20180916'

    Data = []
    Labels = []

    # for file in os.listdir(path):
    for i in range(440):
        file = str(i) + '.jpg'
        file_path_raw = os.path.join(path, file)
        
        # torch :RGB,CHW
        # PIL: RGB, HWC
        # cv2: BGR, HWC
        img_obj = PIL.Image.open(file_path_raw)
        img_obj = img_obj.resize(re_size, PIL.Image.ANTIALIAS)

        img = np.array(img_obj, dtype='float32')
                
        if is_std:
            # img[:, :, 1] = img[:, :, 0]
            # img[:, :, 2] = img[:, :, 0]
            img = img / 255
            # img = img - 0.5

        Data.append(img)
        Labels.append(0)
        
    Data = np.array(Data)
    Labels = np.array(Labels)
    return Data, Labels

