import pika
import numpy as np
import base64
import cv2
from PIL import Image
import io

connection = pika.BlockingConnection(pika.ConnectionParameters('127.0.0.1'))
channel = connection.channel()

channel.queue_declare(queue='hello')

def callback(ch, method, properties, body):
    encoded_data = body
    img_data = base64.b64decode(str(encoded_data))
    image = Image.open(io.BytesIO(img_data))
    img = cv2.cvtColor(np.array(image), cv2.IMREAD_COLOR)
    print('[Worker] Recieve image shape is {}'.format(img.shape))

channel.basic_consume(callback,
                      queue='hello',
                      no_ack=True)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
