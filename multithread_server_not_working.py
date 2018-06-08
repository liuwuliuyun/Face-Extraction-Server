import queue
import sys
import cv2
import numpy as np
import time
import threading
import os
import io
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, Response
import jsonpickle
import base64
from PIL import Image


threads = []
app = Flask(__name__)

q = queue.Queue()

def worker():
    print('Worker!',file=sys.stdout)
    while True:
        item = q.get()
        if item is None:
            continue
        do_work(item)
        q.task_done()

def do_work(item):
    print('Image shape is {}, Image name is {}'.format(item[0].shape,item[1]),file=sys.stderr)
    time.sleep(1)

@app.route('/cfdserver/exfeature', methods=['POST'])
def exfeature():
    global q
    r=request.form
    encoded_data = r['img']
    img_data = base64.b64decode(str(encoded_data))
    image = Image.open(io.BytesIO(img_data))
    img = cv2.cvtColor(np.array(image), cv2.IMREAD_COLOR)
    keyname = r['keyname']
    item = (img,keyname)
    q.put(item)
    print('Current queue length is %d'%(q.qsize())) 
    response = {'message': 'image received. size={}'.format(img.shape),'device':keyname}
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")

if __name__ == "__main__":
    num_worker_threads = 2
    for i in range(num_worker_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)
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

    # stop workers
    for i in range(num_worker_threads):
        q.put(None)
    for t in threads:
        t.join()
