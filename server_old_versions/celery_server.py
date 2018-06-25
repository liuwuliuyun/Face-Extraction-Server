import queue
import sys
import cv2
import numpy as np
import time
import os
import io
import logging
import tensorflow as tf
import time
import random
import base64
import jsonpickle
from ext.extractor import extractor
from ext.aligner import aligner
from celery import Celery, Task
from flask import Flask, request, Response
from PIL import Image


#global flask and celery implementation
app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'amqp://yliu:yliu@localhost:5672/yliu_celery_host'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


#global queue
q = queue.Queue()

class Worker_0_Task(Task):

    _device = None
    _session = None
    _extractor = None
    _aligner = None

    def __init__(self):
        self._device_0 = '/gpu:0'
        self._device_1 = '/gpu:1'
        if self._session is None:
            config = tf.ConfigProto()
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
            self._session = tf.Session(config = config)

        if self._extractor is None:
            self._extractor = extractor(self._session, [self._device_0], 1)
        
        if self._aligner is None:
            self._aligner = aligner(self._session, [self._device_1], 1)


    @property
    def session(self):
        return self._session
    
    @property
    def Extractor(self):
        return self._extractor

    @property
    def Aligner(self):
        return self._aligner



@celery.task(base = Worker_0_Task)
def worker_0():
    global q
    item = q.get()
    print('Image shape is {}, Image name is {}'.format(item[0].shape,item[1]),file=sys.stderr)
    image = item[0]
    image = np.stack([image],axis=0)
    image = worker_0.Aligner.align(image)
    features = worker_0.Extractor.extract(image)
    print('Message From Worker_0 :Length of features is {}'.format(len(features)))


@app.route('/cfdserver/exfeature', methods=['POST'])
def exfeature():
    global q
    r=request.form
    #decode image file
    encoded_data = r['img']
    img_data = base64.b64decode(str(encoded_data))
    image = Image.open(io.BytesIO(img_data))
    img = cv2.cvtColor(np.array(image), cv2.IMREAD_COLOR)

    keyname = r['keyname']

    item = (img,keyname)
    q.put(item)

    print('Current queue length is %d'%(q.qsize()),file=sys.stdout)
    worker_0() 
    response = {'message': 'image received. size={}'.format(img.shape),'device':keyname}
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)
    '''
    im_name_list = os.listdir(im_path)
    
    for im_file in im_name_list:
        im_complete_file = os.path.join(im_path,im_file)
        item = (cv2.imread(im_complete_file),im_file)
        q.put(item)
    '''
    # block until all tasks are done
    q.join()
