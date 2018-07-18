#!/usr/bin/env python

import tensorflow as tf
import argparse
import numpy as np
import cv2
import time
from faceExtractor.extractor import extractor
from faceDetector.aligner import aligner
from faceDetector.detector import detector
from util.queueloader import queueloader
import os
import random
import Queue
import time
import threading

def worker(device, queue_out):
    
    config = tf.ConfigProto() 
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    
    detector_ = detector(session, [device])
    aligner_ = aligner()
    extractor_ = extractor(session,[device], 1)
    
    loader_ = queueloader(img_queue)
    count = 0
    mtcnn_average = 0
    arcface_average = 0
    headnum = 0
    align_average = 0
    try:
        for k in range(14):
            start = time.time()
            images,all_boxes,landmarks = detector_.detect_face(loader_)
            end = time.time()
            mtcnn_average+=end-start
            print("mtcnn is {}".format(end-start))
            for i in range(len(images)):
                if all_boxes[i] is None or landmarks[i] is None:
                    continue
                else:
                    start = time.time()
                    faces = aligner_.align(images[i], all_boxes[i], landmarks[i])
                    end = time.time()
                    align_average+=end-start
                    print("align is {}".format(end-start))
                    start = time.time()
                    features = extractor_.extract(faces)
                    end = time.time()
                    arcface_average+=end-start
                    print("arcface is {}".format(end-start))
                    print("head count is {}".format(len(features)))
                    headnum+=len(features)
                    image = images[i]
                    for bbox in all_boxes[i]:
                        cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
                        cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
                    cv2.imwrite("../testimg/result/%s.jpg" %(str(k).zfill(3)) ,image)
                    for feature in features:
                        queue_out.append(feature)
                    count+=1
        print('mtcnn_average is {}'.format((mtcnn_average+align_average)/14))
        print('arcface_average is {}'.format(arcface_average/14))
        print('headcount_average is {}'.format(headnum/14))
    except KeyboardInterrupt: 
        return

        
queue_out = []
img_queue = Queue.Queue(14)
for i in range(1, 15):
    img_queue.put(cv2.imread('../testimg/all/img%s.jpg' % str(i).zfill(3)))
worker('/cpu:0', queue_out)

'''
for i in range(len(queue_out)):
    for j in range(len(queue_out)):
        if i<j:
            print('image {} and {} similarity is {}'.format(i,j,np.dot(queue_out[i],queue_out[j])))
'''
