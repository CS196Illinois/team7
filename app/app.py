import os
from flask import Flask, render_template, request
import tensorflow as tf
# YOU NEED TF 2.1.0 and PROTOBUF 3.11.3!!!

app = Flask('image_optimizer')


def get_model():
    model = tf.keras.models.load_model('model_ver2.h5')
    return model


def run_model(model, files):
    tensor = []
    tensor[0] = tf.keras.preprocessing.image.img_to_array(files, "channels_last", tf.float16)
    #cropped = tf.image.crop_and_resize(tensor, )
    
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
        model = get_model()  # you can use pickle to load the trained model
        img = request.files['img']
        # year = request.form['year']
        #pred = model.predict()
        # image_scores = run_model(model, request.files)
        arr = []
        return render_template('result.html', images=img)


app.run("localhost", 9999, debug=True)
