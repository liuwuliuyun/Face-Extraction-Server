import pika
import base64
imgfile = '/root/server/testimg/img004.jpg'
with open(imgfile,"rb")as image_file:
    encoded_string = base64.b64encode(image_file.read())

connection = pika.BlockingConnection(pika.ConnectionParameters('127.0.0.1'))
channel = connection.channel()

channel.queue_declare(queue='hello', durable = True)

channel.basic_publish(exchange='',
                      routing_key='hello',
                      body=encoded_string)
print(" [Client] Image send to server")
connection.close()
