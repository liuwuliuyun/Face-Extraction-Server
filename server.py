import queue
import cv2
import numpy as np
import time
import threading
import os


def worker():
    while True:
        item = q.get()
        if item is None:
            break
        do_work(item)
        q.task_done()

def do_work(item):
    print('Image shape is {}, Image name is {}'.format(item[0].shape,item[1]))
    time.sleep(1)

if __name__ == "__main__":
    q = queue.Queue()
    threads = []
    num_worker_threads = 20
    im_path = '/root/server/testimg/'
    for i in range(num_worker_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)
    im_name_list = os.listdir(im_path)
    
    for im_file in im_name_list:
        im_complete_file = os.path.join(im_path,im_file)
        item = (cv2.imread(im_complete_file),im_file)
        q.put(item)

    # block until all tasks are done
    q.join()

    # stop workers
    for i in range(num_worker_threads):
        q.put(None)
    for t in threads:
        t.join()
