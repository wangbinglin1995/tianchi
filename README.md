本代码适合小白，作为入门参考，大神请绕开，别脏了眼（第一赛季线上最高94.3%，最终排名93名）

#tianchi  广东工业智造算法赛 - 铝型材表面瑕疵识别 - 第一赛季：
https://tianchi.aliyun.com/competition/information.htm?spm=5176.11165320.5678.2.49d64443o5raea&raceId=231682 

团队：国立新屋熊


程序说明：
    操作系统：win10 或 Ubuntu16.04，
    python版本：python3
    框架：Keras (基于tensorflow)
    依赖库： tensorflow, Keras, opencv, pillow, sklearn, scipy, numpy, matplotlib
    
    使用的模型为ImageNet预训练的densenet169 (Keras自带)
    参考文献： https://arxiv.org/pdf/1608.06993.pdf
  
    整体思路：数据增强之后直接将网络全部层一起训练,使用经过Imagenet预训练的模型densenet169进行迁移学习
        分别对不同图片大小(img_size分别为：299,400,520)的图片构建多个模型，最后进行模型融合(后验概率求和)
    
        
    数据集整理: 擦花20180901141803对照样本 为凸粉
                擦花20180901141824对照样本 为凸粉
                擦花20180830164545对照样本 为桔皮                
                擦花20180831160713对照样本 不确定，删除
                擦花20180906093612对照样本 不确定，删除

    
代码运行方式：    
    python3 predict.py

    注：predict 函数包含训练/测试 2个过程，
        训练过程需要Keras自带的，Imagenet预训练的 densenet-169 模型（过程需联网）
        测试/预测过程请直接利用“models”文件夹中保存的：*.h5 模型
        如果要测试(predict), 修改 is_train = False (默认)
        如果要重新训练， 请修改 is_train = True
        
        例如，如果要利用训练好的model对test样本进行预测，直接：python3 predict.py
  
  

提交代码文件夹结构：
    project
    |--README.md
    |--data   # 请把download的数据集直接解压到此路径
        |-- guangdong_round1_train2_20180916
        |-- guangdong_round1_test_a_20180916
        |-- guangdong_round1_test_b_20181009
    |--code
        |-- predict.py  主函数，包含：训练/测试 2个过程
        |-- train.py  训练函数：模型构建、模型训练
        |-- data_input.py  数据处理函数，用于导入训练/测试数据
    |--submit
        |-- *.csv  提交的结果
    |--models
        |-- *.h5  保存训练好的模型，用于预测


