import io
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import base64
import os
from flask import Flask, render_template, request, send_from_directory
#import optimizer

# YOU NEED TF 2.1.0, CV2 4.2.0.34, and PROTOBUF 3.11.3!!!
# You also need pillow and opencv-python
# RUN IT INSIDE APP DIRECTORY
from image_augment import ImageAugment

app = Flask('image_optimizer')

UPLOAD_FOLDER = r'E:\Projects\team7\app'
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def get_model():
    model = tf.keras.models.load_model('model_pruned_final.h5')
    return model


# returns a list of image paths (original images)
def get_images(files):
    """converts request.files to an array of image paths"""
    arr = []
    for key, value in files.items():
        value.save("./uploads/"+str(value.filename))

        # filestr = value.read()
        # npimg = np.fromstring(filestr, np.uint8)
        # image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        # arr.append(image)
        arr.append("./uploads/"+str(value.filename))
    return arr


def create_tensor(numpy_images):
    """converts an array of numpy images to a single
    4D tensor that is ready to be handled directly by the model"""
    tensor = []
    boxes = []
    for image in numpy_images:
        # img_tensor = tf.keras.preprocessing.image.img_to_array(image)
        tensor.append(image)
        center_x = len(image[0])/2
        center_y = len(image)/2
        distance = min(center_x, center_y)
        y1 = center_y - distance
        x1 = center_x - distance
        y2 = center_y + distance
        x2 = center_x + distance
        row = [y1, x1, y2, x2]
        boxes.append(row)
    boxes_ind = range(len(tensor))
    crop_size = [150, 150]
    return tf.image.crop_and_resize(tensor, boxes, boxes_ind, crop_size)


# (points 2, 4, 6, 7)
def run_model(image_paths, model):
    image_list = []
    paths = []
    for path in image_paths:
        image = Image.open(path)

        augmenter = ImageAugment(image, path)
        augmented_images = augmenter.augment()
        augmented_paths = augmenter.get_paths()

        # add to paths
        paths.append(path)
        paths.extend(augmented_paths)

        # add to images
        image_list.append(np.array(image))
        image_list.extend(augmented_images)

    # this line is a mess, feel free to break it down into multiple lines for readability
    scores = model.predict_on_batch(create_tensor(np.asarray(image_list)))

    return generate_dict(paths, image_list, scores)


def generate_dict(paths, arrays, scores):
    image_dicts = []
    for index in range(len(paths)):
        image_dicts.append({'path': paths[index], 'np_array': arrays[index], 'score': scores[index]})

    # top 5 images:
    sorted_dict = sorted(image_dicts, key=lambda i: i['score'], reverse=True)
    return sorted_dict[:5]


def save_images(images):
    """ saves PIL images as .jpg's to be used by result.html """
    # TODO: Replace code in this function with code to save numpy images maybe?

    for image in images:
        tf.keras.preprocessing.image.save_img()

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


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(r'E:\Projects\team7\app\uploads', filename)


@app.route('/', methods=['POST'])
def results():
    print(request.files.getlist('img'))
    if request.method == 'POST':
        # write your function that loads the model
        model = get_model()  # you can use pickle to load the trained model
        image_paths = get_images(request.files)
        top5 = run_model(image_paths, model)
        # tensor = create_tensor(images)
        #images = save_images(tensor) # augmentation hasn't run yet this won't run the model
        # year = request.form['year']
        # pred = model.predict()
        return render_template('result.html', pred=123)  # returns nothing useful for now


app.run("localhost", 9999, debug=True)

