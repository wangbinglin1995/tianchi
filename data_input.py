import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt

import scipy.io as sio
import cv2
import os
import random
import PIL
from PIL import ImageOps
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
# from PIL import Autocontrast


def get_train_data(re_size):
    h, w, c = re_size
    re_size = (h, w)

    path_N = r'../data/guangdong_round1_train2_20180916/无瑕疵样本'
    path_1 = r'../data/guangdong_round1_train2_20180916/瑕疵样本/不导电'
    path_2 = r'../data/guangdong_round1_train2_20180916/瑕疵样本/擦花'
    path_3 = r'../data/guangdong_round1_train2_20180916/瑕疵样本/横条压凹'
    path_4 = r'../data/guangdong_round1_train2_20180916/瑕疵样本/桔皮'
    path_5 = r'../data/guangdong_round1_train2_20180916/瑕疵样本/漏底'
    path_6 = r'../data/guangdong_round1_train2_20180916/瑕疵样本/碰伤'
    path_7 = r'../data/guangdong_round1_train2_20180916/瑕疵样本/起坑'
    path_8 = r'../data/guangdong_round1_train2_20180916/瑕疵样本/凸粉'
    path_9 = r'../data/guangdong_round1_train2_20180916/瑕疵样本/涂层开裂'
    path_10 = r'../data/guangdong_round1_train2_20180916/瑕疵样本/脏点'
    path_11 = r'../data/guangdong_round1_train2_20180916/瑕疵样本/其他/'
    path_12 = r'../data/guangdong_round1_train2_20180916/瑕疵样本/其他/划伤'
    path_13 = r'../data/guangdong_round1_train2_20180916/瑕疵样本/其他/碰凹'
    path_14 = r'../data/guangdong_round1_train2_20180916/瑕疵样本/其他/打白点'
    path_15 = r'../data/guangdong_round1_train2_20180916/瑕疵样本/其他/铝屑'
    path_16 = r'../data/guangdong_round1_train2_20180916/瑕疵样本/其他/气泡'
    path_17 = r'../data/guangdong_round1_train2_20180916/瑕疵样本/其他/纹粗'
    path_18 = r'../data/guangdong_round1_train2_20180916/瑕疵样本/其他/粘接'

    paths = [path_N, path_1, path_2, path_3, path_4, path_5,
             path_6, path_7, path_8, path_9, path_10, path_11,
             path_12, path_13, path_14, path_15, path_16, path_17, path_18]

    Data = []
    Labels = []
    for i in range(len(paths)):
    
        if i != 11:
            for file in os.listdir(paths[i]):
                if file == '.DS_Store':
                    continue
                file_path_raw = os.path.join(paths[i], file)
                
                # torch :RGB,CHW(channel, height, width)
                # PIL: RGB, HWC
                # cv2: BGR, HWC
                img_obj = PIL.Image.open(file_path_raw)
                img_obj = img_obj.resize(re_size, PIL.Image.ANTIALIAS)
               
                # 原图
                img = np.array(img_obj, dtype='float32')
                img = img / 255
                Data.append(img)
                Labels.append(i)
                
                # 加噪声            
                # gass = np.abs(np.round(np.random.normal(20, 20, (h,w))))
                # gass = gass.astype(np.uint8)  
                # 直方图均衡
                # img_obj2 = ImageOps.autocontrast(img_obj)
        
        elif i == 11:
        
            for file in os.listdir(paths[i]):
                file_path_raw_sub = os.path.join(paths[i], file)
                if file == '.DS_Store':
                    continue
                if len(os.listdir(file_path_raw_sub)) >= 9:
                    continue
                for imgpath in os.listdir(file_path_raw_sub):
                    
                    if imgpath == '.DS_Store':
                        continue
                    file_path_raw = os.path.join(file_path_raw_sub, imgpath)
                                
                    # torch :RGB,CHW(channel, height, width)
                    # PIL: RGB, HWC
                    # cv2: BGR, HWC
                    img_obj = PIL.Image.open(file_path_raw)
                    img_obj = img_obj.resize(re_size, PIL.Image.ANTIALIAS)
                   
                    # 原图
                    img = np.array(img_obj, dtype='float32')
                    img = img / 255
                    Data.append(img)
                    Labels.append(i)
                    
                    # 加噪声            
                    # gass = np.abs(np.round(np.random.normal(20, 20, (h,w))))
                    # gass = gass.astype(np.uint8)  
                    # 直方图均衡
                    #img_obj2 = ImageOps.autocontrast(img_obj)            
            
            

    Data = np.array(Data)
    Labels = np.array(Labels)
    return Data, Labels
                
               
        
def get_test_data(re_size):
    h, w, c = re_size
    re_size = (h, w)
    path = r'../data/guangdong_round1_test_a_20180916'

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
        # img_obj = ImageOps.equalize(img_obj)
        
        img = np.array(img_obj, dtype='float32')        
                
              
        img = img / 255

        Data.append(img)
        Labels.append(0)
        
    Data = np.array(Data)
    Labels = np.array(Labels)
    return Data, Labels


def get_test_data_b(re_size):
    h, w, c = re_size
    re_size = (h, w)
    path = r'../data/guangdong_round1_test_b_20181009'

    Data = []
    Labels = []

    # for file in os.listdir(path):
    for i in range(1000):
        # print(i)
        file = str(i) + '.jpg'
        file_path_raw = os.path.join(path, file)

        # torch :RGB,CHW
        # PIL: RGB, HWC
        # cv2: BGR, HWC
        img_obj = PIL.Image.open(file_path_raw)
        img_obj = img_obj.resize(re_size, PIL.Image.ANTIALIAS)
        # img_obj = ImageOps.equalize(img_obj)

        img = np.array(img_obj, dtype='float32')

        img = img / 255

        Data.append(img)
        Labels.append(0)

    Data = np.array(Data)
    Labels = np.array(Labels)
    return Data, Labels

