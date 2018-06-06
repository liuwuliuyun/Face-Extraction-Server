from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
import io
import json
import base64
from PIL import Image

# Initialize the Flask application
app = Flask(__name__)


# route http posts to this method
@app.route('/cfdserver/exfeature', methods=['POST'])
def exfeature():
    r = request.form
    # convert string of image data to uint8
    encoded_data = r['img']
    img_data = base64.b64decode(str(encoded_data))
    image = Image.open(io.BytesIO(img_data))
    img = cv2.cvtColor(np.array(image), cv2.IMREAD_COLOR)
    # decode image
    keyname = r['keyname']
    # do some fancy processing here....

    # build a response dict to send back to client
    response = {'message': 'image received. size={}'.format(img.shape),'device':keyname}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
