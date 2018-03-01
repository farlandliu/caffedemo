# coding:utf-8
# fork from https://github.com/adilmoujahid/deeplearning-cats-dogs-tutorial
# alse ref: https://www.cnblogs.com/zhonghuasong/p/7469750.html
# to python 3.6
# -----------------------------------
# 2018-2-28 test result: 
#   Too many images, instance was killed due to low memroy.
#   need to divide the images to several parts to push in lmdb.
# -----------------------------------

import os
import glob
import random
import numpy as np

import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

def make_list():
    train_data = [img for img in glob.glob("../data/temp/train/*jpg")]

    #Shuffle train_data
    random.shuffle(train_data)
    content = ''
    for item in train_data:
        content += item + '\n'
    with open('list.txt','w')as f:
        f.writelines(content)
    f.close()

def load_list():
    with  open('list.txt','r') as f:
        data_list = f.readlines()
    f.close()
    data_list1 = [line[:-1] for line in data_list]
    return data_list1

def make_lmdb_batch():
    batch_size = 600
    data_list = load_list()

    train_lmdb = 'train_lmdb'
    validation_lmdb = 'validation_lmdb'

    # os.system('rm -rf  ' + train_lmdb)
    os.system('rm -rf  ' + validation_lmdb)

    # how many batches
    batch_num = len(data_list) // batch_size

    # in_db = lmdb.open(train_lmdb, map_size=int(1e12))
    # in_txn = in_db.begin(write=True)

    # print ('Creating train_lmdb')
    # for x in range(0 , batch_num+1):

    #     for idx, img_path in enumerate(data_list[0 + batch_size*x:batch_size*(x+1)]):
    #         in_idx = idx + x*batch_size
    #         # leave 1/6 images for validation
    #         if in_idx %  6 == 0:
    #             continue
    #         img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #         img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    #         if 'cat' in img_path:
    #             label = 0
    #         else:
    #             label = 1
    #         datum = make_datum(img, label)
    #         keystr = bytes('{:0>5d}'.format(in_idx), encoding='utf-8')
    #         value = datum.SerializeToString()
    #         in_txn.put(keystr, value)
    #         print (str('{:0>5d}'.format(in_idx)) + ':' + img_path)
    #     # import pdb;pdb.set_trace()
    #     # commit to db
    #     in_txn.commit()
    #     in_txn = in_db.begin(write=True)
    
    in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
    in_txn = in_db.begin(write=True)   

    print ('Creating validation_lmdb')
    for x in range(0 , batch_num+1):

        for idx, img_path in enumerate(data_list[0 + batch_size*x:batch_size*(x+1)]):
            in_idx = idx + x*batch_size
            # leave 1/6 images for validation
            if in_idx % 6 != 0:
                continue
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
            if 'cat' in img_path:
                label = 0
            else:
                label = 1
            datum = make_datum(img, label)
            keystr = bytes('{:0>5d}'.format(in_idx), encoding='utf-8')
            value = datum.SerializeToString()
            in_txn.put(keystr, value)
            print (str('{:0>5d}'.format(in_idx)) + ':' + img_path)
        # import pdb;pdb.set_trace()
        # commit to db
        in_txn.commit()
        in_txn = in_db.begin(write=True)


def make_lmdb():
    train_lmdb = 'train_lmdb'
    validation_lmdb = 'validation_lmdb'

    os.system('rm -rf  ' + train_lmdb)
    os.system('rm -rf  ' + validation_lmdb)


    train_data = [img for img in glob.glob("../data/temp/train/*jpg")]
    #for test and output test results
    # test_data = [img for img in glob.glob("../data/temp/test/*jpg")]

    #Shuffle train_data
    random.shuffle(train_data)

    print ('Creating train_lmdb')

    in_db = lmdb.open(train_lmdb, map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for in_idx, img_path in enumerate(train_data[:100]):
            # leave 1/6 images for validation
            if in_idx %  6 == 0:
                continue
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
            if 'cat' in img_path:
                label = 0
            else:
                label = 1
            datum = make_datum(img, label)
            keystr = bytes('{:0>5d}'.format(in_idx), encoding='utf-8')
            value = datum.SerializeToString()
            in_txn.put(keystr, value)
            print (str('{:0>5d}'.format(in_idx)) + ':' + img_path)
    in_db.close()


    print ('\nCreating validation_lmdb')

    in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for in_idx, img_path in enumerate(train_data[:100]):
            if in_idx % 6 != 0:
                continue
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
            if 'cat' in img_path:
                label = 0
            else:
                label = 1
            datum = make_datum(img, label)
            keystr = bytes('{:0>5d}'.format(in_idx), encoding='utf-8')
            value = datum.SerializeToString()
            in_txn.put(keystr, value)
            print ('{:0>5d}'.format(in_idx) + ':' + img_path)
    in_db.close()

    print ('\nFinished processing all images')

def read_lmdb():
    lmdb_env = lmdb.open('train_lmdb')
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()

    for key, value in lmdb_cursor:
        datum.ParseFromString(value)

        label = datum.label
        #print(str(label))
        import pdb;pdb.set_trace()
        data = caffe.io.datum_to_array(datum)

        #CxHxW to HxWxC in cv2
        image = np.transpose(data, (1,2,0))
        cv2.imwrite(key.decode()+'.jpg', image)
        #cv2.imshow('cv2', image)
        # cv2.waitKey(1)
        print('{},{}'.format(key, label))

if __name__ == "__main__":
    make_lmdb()
