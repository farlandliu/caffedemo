#!/usr/bin/env python
#coding:utf-8
#from http://blog.csdn.net/u011762313/article/details/49851795

##### desc ##########
#export parameters in caffe trained model  
#by liu:2018-2-13
###################
import caffe
import numpy as np
# 使输出的参数完全显示
# 若没有这一句，因为参数太多，中间会以省略号“……”的形式代替
np.set_printoptions(threshold='nan')

# deploy文件
MODEL_FILE = 'cifar10/cifar10_full_train_test.prototxt'
# 预先训练好的caffe模型
PRETRAIN_FILE = 'cifar10/cifar10_full_iter_60000.caffemodel.h5'

# 保存参数的文件
params_txt = 'params.txt'
pf = open(params_txt, 'w')

# 让caffe以测试模式读取网络参数
net = caffe.Net(MODEL_FILE, PRETRAIN_FILE, caffe.TEST)

# 遍历每一层
for param_name in net.params.keys():
    # 权重参数
    weight = net.params[param_name][0].data
    # 偏置参数
    bias = net.params[param_name][1].data

    # 该层在prototxt文件中对应“top”的名称
    pf.write(param_name)
    pf.write('\n')

    # 写权重参数
    pf.write('\n' + param_name + '_weight:\n\n')
    # 权重参数是多维数组，为了方便输出，转为单列数组
    weight.shape = (-1, 1)

    for w in weight:
        pf.write('%ff, ' % w)

    # 写偏置参数
    pf.write('\n\n' + param_name + '_bias:\n\n')
    # 偏置参数是多维数组，为了方便输出，转为单列数组
    bias.shape = (-1, 1)
    for b in bias:
        pf.write('%ff, ' % b)

    pf.write('\n\n')

pf.close