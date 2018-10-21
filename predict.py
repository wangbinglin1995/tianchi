"""
主导训练、预测/测试阶段的主函数

王柄淋，林雪峰，2018-10-09
"""
import data_input
import numpy as np
from train import cnn_densenet
import datetime


# 如果要复现提交结果，请直接使用models文件夹训练好的模型，
# 设置 is_train = False

# 如果要重新训练，设置 is_train = True

is_train = False  

# ======================================================================
# 1. 训练阶段（重新训练模型）
# ======================================================================
if is_train:
    # 多模型融合

    # model 1:----------------------------------------
    cnn1 = cnn_densenet(299, 16, 48, False)
    model1, p1 = cnn1.train()
    x_pre, y_test = cnn1.get_data_for_keras(cnn1.d_predict)
    y1 = cnn1.predict_2(model1, x_pre)

    # model 2: ----------------------------------------
    cnn1 = cnn_densenet(400, 8, 48, False)
    model1, p2 = cnn1.train()
    x_pre, y_test = cnn1.get_data_for_keras(cnn1.d_predict)
    y2 = cnn1.predict_2(model1, x_pre)

    # model 3 :-------------------------------------------------
    cnn1 = cnn_densenet(520, 8, 48, False)
    model1, p3 = cnn1.train()
    x_pre, y_test = cnn1.get_data_for_keras(cnn1.d_predict)
    y3 = cnn1.predict_2(model1, x_pre)

    # 多模型得到最终结果：--------------------
    y_pred = y1 + y2 + y3
    y = np.argmax(y_pred, 1)

    # 写入csv ------------------------------
    filepath = "../submit/submit_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"
    fo = open(filepath, "a")
    for i in range(y.shape[0]):
        if y[i] == 0:
            fo.write(str(i) + '.jpg,norm')
        elif y[i] > 11:
            fo.write(str(i) + '.jpg,defect11')
        else:
            fo.write(str(i) + '.jpg,defect' + str(y[i]))
        fo.write('\n')
    fo.close()

# ===================================================================
# 2. 利用已有的模型进行预测 
# ====================================================================
else:
    # model 1 ： -----------------------------------------
    print('model 1')
    cnn1 = cnn_densenet(299, 1, 10, False)    
    model1 = cnn1.get_model()
    model1.load_weights('../models/a_299_9561.h5')
    d_predict = data_input.get_test_data_b(cnn1.input_shape)
    x_pre, _ = cnn1.get_data_for_keras(d_predict)
    y1 = cnn1.predict_2(model1, x_pre)

   
    # model 2: -------------------------------------
    print('model 2')
    cnn1 = cnn_densenet(400, 1, 12, False)    
    model1 = cnn1.get_model()
    model1.load_weights('../models/a_400_9561.h5')
    d_predict = data_input.get_test_data_b(cnn1.input_shape)
    x_pre, _ = cnn1.get_data_for_keras(d_predict)
    y2 = cnn1.predict_2(model1, x_pre)

    # model 3 : ---------------------------------
    print('model 3')
    # cnn1 = cnn_densenet(400, 1, 12, False)    
    # model1 = cnn1.get_model()
    model1.load_weights('../models/a_400_9610.h5')
    # d_predict = data_input.get_test_data_b(cnn1.input_shape)
    # x_pre, _ = cnn1.get_data_for_keras(d_predict)
    y3 = cnn1.predict_2(model1, x_pre)

    # model 4: ----------------------------------
    print('model 4')
    # cnn1 = cnn_densenet(400, 1, 12, False)    
    # model1 = cnn1.get_model()
    model1.load_weights('../models/a_400_9659.h5')
    # d_predict = data_input.get_test_data_b(cnn1.input_shape)
    # x_pre, _ = cnn1.get_data_for_keras(d_predict)
    y4 = cnn1.predict_2(model1, x_pre)
    
    y5 = 0
    # model 5: -----------------------------------------
    print('model 5')
    cnn1 = cnn_densenet(520, 1, 18, False)
    model1 = cnn1.get_model()
    model1.load_weights('../models/a_520_9561.h5')
    d_predict = data_input.get_test_data_b(cnn1.input_shape)
    x_pre, _ = cnn1.get_data_for_keras(d_predict)
    y5 = cnn1.predict_2(model1, x_pre)

    # 多模型融合, 得到最终结果, ------------------------
    y_pred = y1 + y2 + y3 + y4 + y5
    y = np.argmax(y_pred, 1)

    # 并写入csv -----------------------------
    filepath = "../submit/submit_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"
    fo = open(filepath, "a")
    for i in range(y.shape[0]):
        if y[i] == 0:
            fo.write(str(i) + '.jpg,norm') 
        elif y[i] > 11:   
            fo.write(str(i) + '.jpg,defect11')
        else:
            fo.write(str(i) + '.jpg,defect' + str(y[i]))
        fo.write('\n')        
    fo.close()
