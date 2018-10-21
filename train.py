from __future__ import print_function
from keras.backend.tensorflow_backend import set_session
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras import backend as K
from keras.utils import plot_model

from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, GlobalAveragePooling2D

from keras.initializers import glorot_uniform
from keras.preprocessing import image
from keras.models import Model

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
import scipy.io as sio
import random

import data_input
import datetime


# # # --------------  using GPU with dynamic GPU memory  ----------------------------
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}


class cnn_densenet():
    def __init__(self, img_size, batch_size=8, epochs=25, is_plot=False):
        self.num_classes = 19
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = 0.0006
        self.input_shape = (img_size, img_size, 3)
        self.is_plot = is_plot
  
        self.model_type = keras.applications.densenet.DenseNet169
        # self.model_type = keras.applications.xception.Xception

    def load_data(self):
        """
        获得数据：训练集、验证集、测试集
        """
        Data, Labels = data_input.get_train_data(self.input_shape)
        
        Data, Labels = shuffle(Data, Labels.reshape((-1)))
        train_num = round(Data.shape[0] * 0.9038)

        # Use argument load to distinguish training and testing ----------------------
        d_train_d = Data[:train_num, :, :, :]
        d_train_l = Labels[:train_num]
        
        d_train = d_train_d, d_train_l
        d_eval = Data[train_num:, :, :, :], Labels[train_num:]

        Dp, Lp = data_input.get_test_data(self.input_shape)
        
        self.d_predict = Dp, Lp
        
        return d_train, d_eval

    def get_data_for_keras(self, Data):
        """
        将原始 numpy 格式的数据转化为Keras格式
        """
        x_train, y_train = Data
        img_rows, img_cols, img_cha = self.input_shape

        if K.image_data_format() == 'channels_first':
            print("i am first")
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols, img_cha)
            self.input_shape = (1, img_rows, img_cols, img_cha)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_cha)
            self.input_shape = (img_rows, img_cols, img_cha)

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')

        x_train = x_train.astype('float32')
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        
        return x_train, y_train
        
    def get_improve_data_gen(self, x_train, y_train):
        """
        通过创建一个ImageDataGenerator，来进行
        数据增强：随机旋转、偏移、镜像等，增加样本/提高数据的多样性
        
        """        
        datagen = keras.preprocessing.image.ImageDataGenerator( #实例化
                         # rescale=1/255,                         
                         #channel_shift_range=10,
                         horizontal_flip=True,
                         vertical_flip=True,
                         shear_range=0.1,
                         rotation_range = 30, #图片随机转动的角度 
                         #width_shift_range = 0.1, #图片水平偏移的幅度 
                         #height_shift_range = 0.1, #图片竖直偏移的幅度 
                         # zoom_range = 0
                  ) #随机放大或缩小


        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(x_train)
        # fits the model on batches with real-time data augmentation:
        gen1 = datagen.flow(x_train, y_train, batch_size=y_train.shape[0])

        return gen1


    def train(self):
        """
        训练，主函数
        """
    
        # 1.1 get imgs and 转化为Keras格式
        d_train, d_eval = self.load_data()        
        x_eval, y_eval = self.get_data_for_keras(d_eval)
        xt, yt = self.get_data_for_keras(d_train)
        
        # 1.2 数据增强：随机旋转、偏移、镜像等，增加样本/提高数据的多样性
        gen1 = self.get_improve_data_gen(xt, yt)
        
        # 2. 构建CNN模型
        model = self.get_model()        
        optimizer = keras.optimizers.SGD(lr=self.lr, decay=self.lr/10, momentum=0.9, nesterov=True)
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

        # 3. 训练,并统计loss、accuracy
        history = []
        for epoch_i in range(self.epochs):
            print(epoch_i)
            x_train, y_train = gen1.next()
            history_i = model.fit(x_train, y_train, 
                          batch_size=self.batch_size,
                          epochs=1, verbose=1,
                          # class_weight=self.cw,
                          validation_data=(x_eval, y_eval))
                       
            history.append([epoch_i,
                            history_i.history['loss'][0],
                            history_i.history['acc'][0],
                            history_i.history['val_loss'][0],
                            history_i.history['val_acc'][0]])
            

        # 4. evaluate
        score = model.evaluate(x_eval, y_eval, verbose=1)
        print(model.metrics_names)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        # 5. plot训练过程曲线
        if self.is_plot:
            history = np.array(history)
            import matplotlib.pyplot as plt
            plt.plot(history[:, 0], history[:, 1], label='train loss')
            plt.plot(history[:, 0], history[:, 2], label='train acc')
            plt.plot(history[:, 0], history[:, 3], label='test loss')
            plt.plot(history[:, 0], history[:, 4], label='test acc')
            plt.legend(loc='upper right')
            plt.show()
        # 6. 保存训练的模型，供predict应用
        model.save_weights('../models/a_' + str(self.input_shape[0]) + '.h5')
        
        self.test(model, (x_eval, y_eval))
 
        return model, score[1]


    def test(self, model, data):
        x_test, y_test = data
        y_pred = model.predict(x_test, batch_size=self.batch_size)
        # 统计测式结果
        print('-'*30 + 'Begin: test' + '-'*30)
        array = np.argmax(y_pred, 1) == np.argmax(y_test, 1)
        print(np.argmax(y_pred, 1)[np.logical_not(array)])
        print(np.argmax(y_test, 1)[np.logical_not(array)])
        test_acc = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]
        print('Test acc:', test_acc)

        print(array.shape)
        print(x_test.shape)
        print('-' * 30 + 'End: test' + '-' * 30)
        
    def predict(self, model, x_predict):        
        y_pred = model.predict(x_predict, batch_size=self.batch_size)        
        y = np.argmax(y_pred, 1)

        filepath = "../submit/submit_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"
        fo = open(filepath, "a")
        for i in range(y.shape[0]):
            print(i, y[i])
            if y[i] == 0:
                fo.write(str(i) + '.jpg,norm') 
            elif y[i] > 11:   
                fo.write(str(i) + '.jpg,defect11')
            else:
                fo.write(str(i) + '.jpg,defect' + str(y[i]))
            fo.write('\n')        
        fo.close()
        
    def predict_2(self, model, x_predict):        
        y_pred = model.predict(x_predict, batch_size=self.batch_size) 
        return y_pred
        
    def get_model(self):
        """
        构建模型
        """
        base_model = self.model_type(include_top=False, weights='imagenet',input_shape=self.input_shape)
        x = base_model.output

        x = Dropout(0.2)(x)

        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x=Dropout(0.1)(x)

        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model


if __name__ == '__main__':

    # 模型融合：3个模型加权/相加
    
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
