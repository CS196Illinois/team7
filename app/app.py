from flask import Flask, render_template, request

app = Flask('image_optimizer')

from types import SimpleNamespace


def dummy():
    return 123


def get_model():
    model = SimpleNamespace(predict=dummy)
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
        image_scores = run_model(model, request.files)
        return render_template('result.html', pred=image_scores)


app.run("localhost", 9999, debug=True)
