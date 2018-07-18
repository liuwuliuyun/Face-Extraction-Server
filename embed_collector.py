import zmq
import os
import sys
import numpy as np

context = zmq.Context()
results_receiver = context.socket(zmq.PULL)
results_receiver.bind('tcp://127.0.0.1:5558')

if __name__=='__main__':
    while True:
        try:
            result = results_receiver.recv()
            key = result[0:8]
            embedding = result[8:]
            embedding = np.fromstring(embedding, dtype = np.float32)
            embedding = np.reshape(embedding,(512,1))
            print('Receive Successful\nMessage Key is {} \nShape is {}'.format(str(key),embedding.shape))
        except KeyboardInterrupt:
            print('Interrupted by Root')
            break
        except:
            print('Receive Failed')
