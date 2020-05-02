import io
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import base64
import os
from flask import Flask, render_template, request
import optimizer

# YOU NEED TF 2.1.0, CV2 4.2.0.34, and PROTOBUF 3.11.3!!!
# You also need pillow and opencv-python
from image_augment import ImageAugment

app = Flask('image_optimizer')


def get_model():
    model = tf.keras.models.load_model('/Users/ishita/team7/app/model_ver2.h5')
    return model


# THIS FUNCTION USES PLACEHOLDERS: don't expect this to work right now
def run_model(model, files):
    for image in files:
        augmenter = ImageAugment(image)
        augmenter.augment()

    augmented_images = model.augment(files)
    scores = model.calculate_socres(augmented_images)  # list or a dict
    best_images = {}
    for index in range(scores.len):
        best_images.update({augmented_images.get(index): scores[index]})

    return best_images


def get_images(files):
    """converts request.files to an array of PIL images"""
    arr = []
    for key, value in files.items():
        filestr = value.read()
        npimg = np.fromstring(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        arr.append(image)
        return arr


def create_tensor(images):
    """converts an array of PIL images to a single
    4D tensor that is ready to be handled directly by the model"""
    tensor = []
    boxes = []
    for image in images:
        img_tensor = tf.keras.preprocessing.image.img_to_array(image)
        tensor.append(img_tensor)
        center_x = len(img_tensor[0])/2
        center_y = len(img_tensor)/2
        distance = min(center_x, center_y)
        y1 = center_y - distance
        x1 = center_x - distance
        y2 = center_y + distance
        x2 = center_x + distance
        row = [y1, x1, y2, x2]
        boxes.append(row)
    boxes_ind = range(len(tensor))
    crop_size = [200, 200]
    return tf.image.crop_and_resize(tensor, boxes, boxes_ind, crop_size)


def save_images(images):
    """ saves PIL images as .jpg's to be used by result.html """
    image_objects = []
    for image in images:
        # convert numpy array to PIL Image
        img = Image.fromarray(image)

        # create file-object in memory
        img_buffer = io.BytesIO()

        # write JPEG in file-object
        img.save(img_buffer, 'JPEG')

        # move to beginning of file
        img_buffer.seek(0)
        img_tag = 'data:image/jpeg;base64,' + base64.b64encode(img_buffer.getvalue()).decode('ascii')
        image_objects.append(img_tag)

    return image_objects


@app.route('/')
def show_index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def results():
    print(request.files.getlist('img'))
    if request.method == 'POST':
        # write your function that loads the model
        images = optimizer.run(request.files)
        # year = request.form['year']
        # pred = model.predict()
        
        return render_template('result.html', images=images)  # returns nothing useful for now


app.run("localhost", 9999, debug=True)
