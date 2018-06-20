#!/usr/bin/env python


import tensorflow as tf
import argparse
import numpy as np
import cv2
import time
from extractor import extractor
from aligner import aligner
import os
import random


def work(device, queue_in, queue_out):
    
    config = tf.ConfigProto() 
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    

    extractor_ = extractor(session,[device], 1)
    aligner_ = aligner(session, [device], 1)

    for image in queue_in:
        image = np.stack([image], axis=0)
        image = aligner_.align(image)
        features = extractor_.extract(image)
        #print(len(features))
        for feature in features:
            queue_out.append(feature)



queue_in = [cv2.imread('data/%d.jpg' % i) for i in range(1,7)]
queue_out = []
work('/cpu:0', queue_in, queue_out)

for i in range(len(queue_out)):
    for j in range(len(queue_out)):
        if i<j:
            print('image {} and {} similarity is {}'.format(i,j,np.dot(queue_out[i],queue_out[j])))


