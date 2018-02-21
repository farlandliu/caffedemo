#!/usr/bin/env python3
# coding:utf-8
# ref caffe example: 00 classification

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import caffe
# display plots in this notebook
# %matplotlib inline

root = '/home/farland/caffedemo/'

def classify():

    # set display defaults
    plt.rcParams['figure.figsize'] = (10, 10)        
    # large images
    plt.rcParams['image.interpolation'] = 'nearest'  
    # don't interpolate: show square pixels
    plt.rcParams['image.cmap'] = 'gray'  
    # use grayscale output rather than a (potentially misleading) color heatmap
    mu = np.load( '../data/cifar10.mean.npy')
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    print ('mean-subtracted values:')
    for x in zip('BGR', mu):
        print(x)

    net = get_net()
    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 32)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

    
    # set the size of the input (we can skip this if we're happy
    #  with the default; we can also change it later, e.g., for different batch sizes)
    net.blobs['data'].reshape(50,        # batch size
                              3,         # 3-channel (BGR) images
                              32, 32)  # image size is 227x227
    image = caffe.io.load_image('../data/images/car.jpg')
    transformed_image = transformer.preprocess('data', image)
    plt.imshow(image)

    #starting classify
    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    output = net.forward()

    output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

    print( 'predicted class is:', output_prob.argmax())

    # load ImageNet labels
    # import pdb;pdb.set_trace()
    labels_file =  '../data/synset_words_cifar10.txt'
    
    labels = np.loadtxt(labels_file, str, delimiter='\t')

    # print ('output label:', labels[output_prob.argmax()])

    # sort top five predictions from softmax output
    top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

    print ('probabilities and labels:')
    for x in zip(output_prob[top_inds], labels[top_inds]):
        print(x)


def get_net():

    caffe.set_mode_cpu()

    model_def = 'cifar10_full.prototxt'
    model_weights = 'cifar10_full_iter_60000.caffemodel.h5'

    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    return net


def mean_npy(fn):
    '''
    http://blog.csdn.net/may0324/article/details/52316967
    binaryproto -> npy

    Tranfer caffe binary mean file to python mean npy file.
    Save npy mean to data folder
    '''
    import numpy as np
    import caffe
    import sys

    mean_file_name = fn[:-12]

    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( fl , 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    out = arr[0]
    np.save( mean_file_name + '.npy' , out )


if __name__ == "__main__":
    classify() 