import  re
import os
from skimage.transform import resize
from PIL import Image

from keras.models import load_model
import base64
import numpy as np
import tensorflow as tf

global model


model = load_model(os.path.join('pkl_objects','handwritten_digit_recognizer','my_model.h5'))
graph = tf.get_default_graph()

def convertImage(imgData1):

    imgstr = re.search(b'base64,(.*)', imgData1).group(1)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))


def digit(imgData):
    convertImage(imgData)

    x=  Image.open('output.png')
    x = np.invert(x)
    x = resize(x,(28,28),order=1, mode='constant', cval=0, clip=False, preserve_range=True)
    x = x.reshape(1,28,28,1)

    with graph.as_default():
        y_pred=model.predict(x)

    for i in range(len(y_pred[0])):

        if y_pred[0][i]==1:
            m=i
    return m
