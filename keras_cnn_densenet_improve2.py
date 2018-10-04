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
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
import scipy.io as sio
import random

import data_input



# # # --------------  using only CPU ,no GPU  -------------------------------------
# num_cores = 7
# config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
#         inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
#         device_count = {'CPU' : 1, 'GPU' : 0})
# set_session(tf.Session(config=config))

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


# ----------------------------------------------------------------------------------------------
# getdata
# ---------------------------------------------------------------------------------------------

class cnn_inception():
    def __init__(self, is_plot=False):
        
        self.num_classes = 12
        self.batch_size = 7
        self.epochs = 14
        self.lr = 0.001
        
        self.is_plot = is_plot
        
        # self.model_type = keras.applications.inception_resnet_v2.InceptionResNetV2
        self.model_type = keras.applications.densenet.DenseNet201
        # self.model_type = keras.applications.nasnet.NASNetMobile
        # self.model_type=keras.applications.inception_v3.InceptionV3
        # self.model_type = keras.applications.xception.Xception

        self.is_std = True
        self.re_size = (299, 299)
        self.input_shape = (299, 299, 3)


    def load_data(self):
        # Data, Labels = data_input.get_train_data(self.re_size, self.is_std)
        # sio.savemat('data_train_299.mat', {'Data': Data, 'Labels': Labels})
        d_tmp = sio.loadmat('data_train_299.mat')
        Data = d_tmp['Data']
        Labels = d_tmp['Labels']

        Data, Labels = shuffle(Data, Labels.reshape((-1)), random_state=0)
        train_num = round(Data.shape[0] * 0.9138)

        # Use argument load to distinguish training and testing ----------------------
        d_train = Data[:train_num, :, :, :], Labels[:train_num]
        d_eval = Data[train_num:, :, :, :], Labels[train_num:]

        # Dp, Lp = data_input.get_test_data(self.re_size, self.is_std)
        # sio.savemat('data_predict_299.mat', {'Data': Dp, 'Labels': Lp})
        d_tmp = sio.loadmat('data_predict_299.mat')
        Dp = d_tmp['Data']
        Lp = d_tmp['Labels'].reshape((-1))

        self.d_predict = Dp, Lp

        return d_train, d_eval

    def get_data(self, Data, is_test=True):
        x_train, y_train = Data
        img_rows, img_cols, img_cha = self.input_shape

        if K.image_data_format() == 'channels_first':
            print("i am first")
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols, img_cha)
            self.input_shape = (1, img_rows, img_cols, img_cha)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_cha)
            self.input_shape = (img_rows, img_cols, img_cha)

        # x_train /= 255
        # x_train -= 0.5

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')

        x_train = x_train.astype('float32')


            # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, self.num_classes)

        if not is_test:
            datagen = keras.preprocessing.image.ImageDataGenerator( #实例化
                         # rescale=1,
                         #channel_shift_range=10,
                         horizontal_flip=False,
                         vertical_flip=False,
                         rotation_range = 30, #图片随机转动的角度 
                         width_shift_range = 0.2, #图片水平偏移的幅度 
                         height_shift_range = 0.2, #图片竖直偏移的幅度 
                         zoom_range = 0.14) #随机放大或缩小


            # compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            datagen.fit(x_train)
            # fits the model on batches with real-time data augmentation:
            gen1 = datagen.flow(x_train, y_train, batch_size=round(y_train.shape[0]/2))

            for i in range(3):
                x_train2, y_train2 = gen1.next()
                # c = x_train2*255
                x_train = np.vstack((x_train, x_train2))
                y_train = np.vstack((y_train, y_train2))

        return x_train, y_train


    def train(self):

        d_train, d_eval = self.load_data()
        x_train, y_train = self.get_data(d_train, False)
        x_eval, y_eval = self.get_data(d_eval, True)


        base_model = self.model_type(include_top=False, weights='imagenet',input_shape=self.input_shape)
        model = base_model

        # add a global spatial average pooling layer
        x = base_model.output        
        
        x = Dropout(0.3)(x)   
        
        x = GlobalAveragePooling2D()(x)
        
        # x=Flatten()(x)
        
        x = Dense(1024, activation='relu')(x)    # let's add a fully-connected layer
        # x=Dropout(0.1)(x)
        
        predictions = Dense(self.num_classes, activation='softmax')(x)    # logistic layer -- we have 12 classes
        model = Model(inputs=base_model.input, outputs=predictions)    # the model we will train

        
        # optimizer='rmsprop'
        optimizer = keras.optimizers.SGD(lr=self.lr, decay=self.lr/10, momentum=0.9, nesterov=True)

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

        history = LossHistory()
        model.fit(x_train, y_train, batch_size=self.batch_size,
                                    epochs=self.epochs, verbose=1,
                                    validation_data=(x_eval, y_eval), callbacks=[history])

        score = model.evaluate(x_eval, y_eval, verbose=1)
        print(model.metrics_names)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        # model.save_weights('a.h5')
        if self.is_plot:
            history.loss_plot('epoch')
        return model


    def test(self, model, data):
        x_test, y_test = data
        y_pred = model.predict(x_test, batch_size=self.batch_size)
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
       
        fo = open("result_xception_improve.csv", "a")

        for i in range(y.shape[0]):
            print(i, y[i])

            if y[i] == 0 or y[i] > 11:
                fo.write(str(i) + '.jpg,norm')            
            else:
                fo.write(str(i) + '.jpg,defect' + str(y[i]))
            fo.write('\n')
        
        fo.close()
        


if __name__ == '__main__':

    cnn1 = cnn_inception(True)
    model = cnn1.train()
    
    # model.load_weights('a.h5')
    x_pre, y_test = cnn1.get_data(cnn1.d_predict)
    cnn1.predict(model, x_pre)
