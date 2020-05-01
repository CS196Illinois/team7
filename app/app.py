import os
import numpy as np
import cv2
from flask import Flask, render_template, request
import tensorflow as tf
# YOU NEED TF 2.1.0, CV2 4.2.0.34, and PROTOBUF 3.11.3!!!
# You also need pillow
app = Flask('image_optimizer')


def get_model():
    model = tf.keras.models.load_model('model_ver2.h5')
    return model

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

def save_images():
    """ saves PIL images as .jpg's to be used by result.html """
    return

def run_model(model, files):
    augmented_images = model.augment(files)
    scores = model.calculate_socres(augmented_images)  # list or a dict
    image_scores = {}
    for index in range(scores.len):
        image_scores.update({augmented_images.get(index): scores[index]})

    return image_scores


@app.route('/')
def show_index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def results():
    print(request.files)
    if request.method == 'POST':
        # write your function that loads the model
        model = get_model() # you can use pickle to load the trained model
        images = get_images(request.files)
        tensor = create_tensor(images)
        # year = request.form['year']
        # pred = model.predict()
        # image_scores = run_model(model, request.files)
        
        return render_template('result.html', images=images) # returns nothing useful for now


app.run("localhost", 9999, debug=True)
