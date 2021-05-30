import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np


model = keras.models.load_model("../nn.h5")

# load the image
from PIL import Image
import io
with open('three.png', 'rb') as file:
    image_bytes = file.read()
    pillow_img = Image.open(io.BytesIO(image_bytes)).convert('L')


# transform image, same as for training!
data = np.asarray(pillow_img)
data = data / 255.0
data = data[np.newaxis, ..., np.newaxis]
# --> [1, x, y, 1]
data = tf.image.resize(data, [28, 28])


# predict
predictions = model(data)
predictions = tf.nn.softmax(predictions)
pred0 = predictions[0]
label0 = np.argmax(pred0)
print(label0)
