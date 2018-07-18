#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import cv2
from faceExtractor.extractor import extractor
from faceDetector.aligner import aligner
from faceDetector.detector import detector
from util.taskloader import taskloader
from util.rsphelper import new_c_response, new_r_response
import random
import zmq
import os
import sys

#arg 
arglen = len(sys.argv)
if arglen == 1:
    print("No ip address, will use 127.0.0.1 ...")
    strIp = "tcp://127.0.0.1"
elif arglen == 2:
    strIp = "tcp://"+sys.argv[1]
else:
    print("Ip seems wrong, will use 127.0.0.1 ...")
    strIp = "tcp://127.0.0.1"

context = zmq.Context()
req_socket = context.socket(zmq.PULL)
rsp_socket = context.socket(zmq.PUSH)

try:
    req_socket.bind(strIp + ":5555")
    rsp_socket.bind(strIp + ":5556")
except:
    print("The faceService cannot start! Please check if there have been a faceService running...")
    exit()

config = tf.ConfigProto() 
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

detector_ = detector(session, ['/cpu:0'])
aligner_ = aligner()
extractor_ = extractor(session,['/cpu:0'], 1)

task_loader_ = taskloader(req_socket)
try:
    while True:
        reqIds, reqTypes, images, all_boxes, landmarks = detector_.detect_face(task_loader_)
        for i in range(len(images)):
            #for counting
            if reqTypes[i] == u'C':
                print('Face counting request:%s...' % reqIds[i])
                if all_boxes[i] is None or landmarks[i] is None:
                    rsp_data = new_c_response(reqIds[i], 0)
                else:
                    rsp_data = new_c_response(reqIds[i], len(all_boxes[i]))
                
                rsp_socket.send_string(rsp_data)
                print('Replied.')

            #for output images
            elif reqTypes[i] == u'D':
                print('Boxing out request:%s...' % reqIds[i])
                if all_boxes[i] is not None:
                    image = images[i]
                    for bbox in all_boxes[i]:
                        cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
                        cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))

                    if not os.path.exists('./testimg'):
                        os.mkdir('./testimg')
                    cv2.imwrite("./testimg/%03d-%04d.jpg" %(i, random.randint(1,9999)) ,image)
                    print('Done.')
                else:
                    print('Cannot find a face in the image.')

            #for recognition
            elif reqTypes[i] == u'R':
                print('Face recognition request:%s...' % reqIds[i])
                if all_boxes[i] is None or landmarks[i] is None:
                    rsp_data = new_r_response(reqIds[i], [])
                else:
                    faces = aligner_.align(images[i], all_boxes[i], landmarks[i])
                    features = extractor_.extract(faces)
                    rsp_data = new_r_response(reqIds[i], features)

                rsp_socket.send_string(rsp_data)
                print('Replied.')

            else:
                continue
                
except KeyboardInterrupt: 
    req_socket.close()
    rsp_socket.close()