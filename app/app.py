from flask import Flask, render_template, request, send_from_directory
import numpy as np
from PIL import Image
import tensorflow as tf

from image_augment import ImageAugment

# RUN IT INSIDE APP DIRECTORY

app = Flask('image_optimizer')

UPLOAD_FOLDER = r'/Users/ishita/team7/app/uploads'  # USERS: CHANGE THIS
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def get_model():
    model = tf.keras.models.load_model('model_ver2.h5')
    return model


# returns a list of image paths (original images)
def get_images(files):
    """converts request.files to an array of image paths"""
    arr = []
    for key, value in files.items():
        value.save("./uploads/"+str(value.filename))
        arr.append("./uploads/"+str(value.filename))
    return arr


# Returns a list of image paths
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

    scores = model.predict_on_batch(create_tensor(np.asarray(image_list))).numpy()

    return fetch_best_images(paths, scores)


def create_tensor(numpy_images):
    """converts an array of numpy images to a single
    4D tensor that is ready to be handled directly by the model"""
    tensor = []
    boxes = []
    for image in numpy_images:
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


# returns a list of image paths
def fetch_best_images(paths, scores):
    image_dicts = []
    for index in range(len(paths)):
        image_dicts.append({'path': '.' + paths[index], 'score': scores[index]})

    # top 5 images:
    sorted_dict = sorted(image_dicts, key=lambda i: i['score'], reverse=True)
    sorted_dict = sorted_dict[:5]

    required_paths = [sub['path'] for sub in sorted_dict]
    return required_paths


@app.route('/')
def show_index():
    return render_template('index.html')


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/', methods=['POST'])
def results():
    if request.method == 'POST':
        # called when user hits submit
        model = get_model()
        image_paths = get_images(request.files)
        top5 = run_model(image_paths, model)
        return render_template('result.html', images=top5)


app.run("localhost", 9999, debug=True)

