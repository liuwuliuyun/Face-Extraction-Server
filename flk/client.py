import requests
import json
import cv2
import base64

addr = 'http://localhost:5000'
test_url = addr + '/cfdserver/exfeature'
keyname = 'device_01'
imgfile = '/root/server/testimg/img004.jpg'

with open(imgfile, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())

data_to_post = {'img':encoded_string,'keyname':keyname}

# send http request with image and receive response
response = requests.post(test_url, data=data_to_post)
# decode response
print(json.loads(response.text))
