import requests
import json
import cv2
import base64
import threading

addr = 'http://localhost:5000'
test_url = addr + '/cfdserver/exfeature'
keyname = 'device_01'
imgfile = '/root/server/testimg/img004.jpg'

with open(imgfile, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())

data_to_post = {'img':encoded_string,'keyname':keyname}

def post_data_to_server(data_to_post, test_utl):
    # send http request with image and receive response
    response = requests.post(test_url, data=data_to_post)
    # decode response
    print('Response from thread {}'.format(threading.get_ident()))
    print(json.loads(response.text))

post_workers = []

for i in range(0,100):
    t = threading.Thread(target=post_data_to_server, args=(data_to_post,test_url))
    t.start()
    post_workers.append(t)

for t in post_workers:
    t.join()
    
