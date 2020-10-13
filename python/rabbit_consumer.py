#coding:utf-8

import os
import pika
import time

def callback(ch, method, properties, body):
    print(" [x] Received %r" % body)
    import time
    time.sleep(10)
    print('ok')
    ch.basic_ack(delivery_tag = method.delivery_tag)

if __name__ == '__main__':
    consumer = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = consumer.channel()
    channel.queue_declare('' , False , True , False , False)

    channel.basic_consume(callback , queue='' , no_ack=False)
    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


