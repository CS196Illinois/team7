from flask import Flask, render_template, request
import tensorflow as tf
# YOU NEED TF 2.1.0 and PROTOBUF 3.11.3!!!
import os

app = Flask('image_optimizer')


def get_model():
    model = tf.keras.models.load_model('model42604.h5')
    return model


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
        model = get_model()  # you can use pickle to load the trained model

        # year = request.form['year']
        # pred = model.predict()
        # image_scores = run_model(model, request.files)
        return render_template('result.html', pred=123)


app.run("localhost", 9999, debug=True)
