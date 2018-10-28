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

from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

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

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure(figsize=(6.3, 4.7))
        plt.rcParams.update(params)
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.subplots_adjust(0.06, 0.1, 0.96, 0.94, 0.2, 0.3)
        plt.savefig("result.png")
        plt.show()


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
        
    def get_improve_data_gen(self, x_train, y_train, is_improve=False):
        """
        通过创建一个 ImageDataGenerator , 来进行: 数据增强
            : 随机旋转、偏移、镜像等，增加样本/提高数据的多样性
        
        注：
          此处创建了一个迭代器，可直接用于 fit_generator，
          也可以不用 ImageDataGenerator, 自己写这个迭代器,           
        
        """  
        if is_improve:
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

        else:
            datagen = keras.preprocessing.image.ImageDataGenerator()

        datagen.fit(x_train)
        # fits the model on batches with real-time data augmentation:
        gen1 = datagen.flow(x_train, y_train, batch_size=self.batch_size)

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
        gen_train = self.get_improve_data_gen(xt, yt, True)
        gen_val   = self.get_improve_data_gen(x_eval, y_eval, False)
        
        # 2. 构建CNN模型
        model = self.get_model() 
        
        logging = TensorBoard(log_dir='../logs')        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
        history = LossHistory()

        # 3. 训练,并统计loss、accuracy
        # history = []
        max_val_acc = 0
        optimizer = keras.optimizers.SGD(lr=self.lr, decay=self.lr/10, momentum=0.9, nesterov=True)
                 
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
                
        model.fit_generator(gen_train, 
                            steps_per_epoch=max(1, xt.shape[0]//self.batch_size),
                            # class_weight=self.cw,
                            validation_data=gen_val,      
                            validation_steps=max(1, x_eval.shape[0]//self.batch_size),
                            epochs=self.epochs,
                            callbacks=[logging, early_stopping, history])
            

        # 4. evaluate
        if self.is_plot:
            history.loss_plot('epoch')
        score = model.evaluate(x_eval, y_eval, verbose=1)
        print(model.metrics_names)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])        
        
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
    
    # 1. train:-----------------------------------------------------------
    cnn1 = cnn_densenet(224, 8, 48, False)
    model1, p1 = cnn1.train()
    
    # 2. predict: ----------------------------------------------------------
    # first: get dataset for predict: guangdong_round1_test_a_20180916 数据集   
    Dp, Lp = data_input.get_test_data_a(cnn1.input_shape)  
    d_predict = Dp, Lp
    # second: 转化为Keras数据格式
    x_pre, y_test = cnn1.get_data_for_keras(d_predict)
    # third： predict using the trained model 
    y_pred = cnn1.predict_2(model1, x_pre)
    y = np.argmax(y_pred, 1)

    # 3. 写入csv -----------------------------------------------------------
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
