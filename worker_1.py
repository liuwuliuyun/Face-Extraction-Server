import tensorflow as tf
import numpy as np
import sys
import time
import pika
import base64
import cv2
import io
from PIL import Image
from ext.extractor import extractor
from ext.aligner import aligner
#TODO set different gpu devices
device = '/gpu:1'

config = tf.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
session = tf.Session(config = config)

Extractor = extractor(session, [device], 1)
Aligner = aligner(session, [device], 1)

connection = pika.BlockingConnection(pika.ConnectionParameters('127.0.0.1'))
channel = connection.channel()
channel.queue_declare(queue='worker_queue',durable=True)

def callback(ch, method, properties, body):
    try:
        start = time.time()
        encoded_data = body
        img_data = base64.b64decode(encoded_data)
        image = Image.open(io.BytesIO(img_data))
        img = cv2.cvtColor(np.array(image),cv2.IMREAD_COLOR)
        image = np.stack([image],axis=0)
        image = Aligner.align(image)
        features = Extractor.extract(image)
        #TODO send features to database to compare
        print('Worker_0: Length of feature is {}\n \t \t Time used is {} s'.format(len(features), time.time()-start))
        #ch.basic_ack(delivery_tag = method.delivery_tag)
    except:
        print('Worker_0: Internal Error Occurred')
        #ch.basic_ack(delivery_tag = method.delivery_tag)
if __name__=='__main__':
    channel.basic_consume(callback,queue='worker_queue',no_ack=True)
    channel.start_consuming()
