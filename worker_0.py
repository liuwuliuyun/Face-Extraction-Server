import tensorflow as tf
import numpy as np
import sys
import time
import base64
import zmq
import cv2
import io
import os
from PIL import Image
from ext.extractor import extractor
from ext.aligner import aligner


#TODO set different gpu devices
device = '/gpu:0'

config = tf.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
session = tf.Session(config = config)

Extractor = extractor(session, [device], 1)
Aligner = aligner(session, [device], 1)

context = zmq.Context()

consumer_receiver = context.socket(zmq.PULL)
consumer_receiver.setsockopt(zmq.SNDHWM, 1000)
consumer_receiver.connect('tcp://127.0.0.1:5557')

consumer_sender = context.socket(zmq.PUSH)
consumer_sender.setsockopt(zmq.RCVHWM, 1000)
consumer_sender.connect('tcp://127.0.0.1:5558')

if __name__=='__main__':

    while True:
        #try:
            start = time.time()
            encoded_data = consumer_receiver.recv_string()
            key = encoded_data[0:9]
            encoded_data = encoded_data[9:]
            img_data = base64.b64decode(encoded_data)
            image = Image.open(io.BytesIO(img_data))
            img = cv2.cvtColor(np.array(image),cv2.IMREAD_COLOR)
            image = np.stack([img],axis=0)
            image = Aligner.align(image)
            features = Extractor.extract(image)
            print('[CFDS WORKER LOG]Worker_0: Length of feature is {}\n \t \t Time used is {} s'.format(len(features), time.time()-start))
            if len(features)>0:
                for feature in features:
                    data_to_dump = key.encode('utf-8')+feature.tostring()
                    consumer_sender.send(data_to_dump)
            
        #except Exception as e:
        #    print(e)
        #    print('[CFDS WORKER LOG]Worker_0: Internal Error Occurred')
        #    break

