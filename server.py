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
import pika
import base64
import jsonpickle
#from multiprocessing import Process, Queue
from threading import Lock
from flask import Flask, request, Response
from PIL import Image
from ext.extractor import extractor
from ext.aligner import aligner

#global flask and celery implementation
app = Flask(__name__)

connection = pika.BlockingConnection(pika.ConnectionParameters('127.0.0.1'))
channel = connection.channel()
channel.queue_declare(queue = 'worker_queue', durable = True)

lock = Lock()

@app.route('/cfdserver/exfeature', methods=['POST'])
def exfeature():
    global channel
    r=request.form
    #decode image file
    encoded_data = r['img']
    keyname = r['keyname']
    #TODO check queue length before publish
    #TODO send keyname with image
    '''
    try:
        channel.basic_publish(exchange='',routing_key='worker_queue',body=encoded_data)
        #TODO discard new image if queue is full
        response = {'message': 'image received successful', 'device':keyname}
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")
    except:
        response = {'message': 'image received failed, server busy, plz retry'}
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=503, mimetype='application/json')
    '''
    with lock:
        channel.basic_publish(exchange='',routing_key='worker_queue',body=encoded_data)

    #TODO discard new image if queue is full
    response = {'message': 'image received successful', 'device':keyname}
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")
    

if __name__ == "__main__":
    connection = pika.BlockingConnection(pika.ConnectionParameters('127.0.0.1'))
    app.run(host="0.0.0.0",port=5000,threaded=True)
