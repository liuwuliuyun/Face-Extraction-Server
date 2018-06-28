import tensorflow as tf
import numpy as np
import sys
import time
import pika
import base64
import cv2
import io
import os
from PIL import Image
from ext.extractor import extractor
from ext.aligner import aligner


#TODO set different gpu devices
device = '/gpu:0'
face_embed_path = '/root/database/evalfeature/'
face_embed_list = []

config = tf.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
session = tf.Session(config = config)

Extractor = extractor(session, [device], 1)
Aligner = aligner(session, [device], 1)

connection = pika.BlockingConnection(pika.ConnectionParameters('127.0.0.1'))
channel = connection.channel()
channel.queue_declare(queue='worker_queue',durable=True)

def get_face_id(features):

    face_id_list = []    

    if len(features)>0:
        for feature in features:
            face_sim = 0
            face_id =''
            for face_embed in face_embed_list:
                temp_sim = np.dot(face_embed[1],feature)
                if temp_sim>face_sim:
                    face_sim = temp_sim
                    face_id = face_embed[0]
            face_id_list.append(face_id)

    return face_id_list
                

def callback(ch, method, properties, body):
    #try:
        start = time.time()
        encoded_data = body
        key = encoded_data[0:9]
        encoded_data = encoded_data[9:]
        img_data = base64.b64decode(encoded_data)
        image = Image.open(io.BytesIO(img_data))
        img = cv2.cvtColor(np.array(image),cv2.IMREAD_COLOR)
        image = np.stack([image],axis=0)
        image = Aligner.align(image)
        features = Extractor.extract(image)
        print('[CFDS WORKER LOG]Worker_0: Length of feature is {}\n \t \t Time used is {} s'.format(len(features), time.time()-start))
        face_id_list = get_face_id(features)
        print(face_id_list)
    #except:
        #print('[CFDS WORKER LOG]Worker_0: Internal Error Occurred')

if __name__=='__main__':
    #initialize face embedding list
    for embed_file in os.listdir(face_embed_path):
        complete_embed_path = os.path.join(face_embed_path,embed_file)
        face_embed_list.append((embed_file.split('.')[0], np.load(complete_embed_path)))

    channel.basic_consume(callback,queue='worker_queue',no_ack=True)
    channel.start_consuming()
