from flask import Flask, render_template, request
from app import optimizer
# YOU NEED TF 2.1.0 and PROTOBUF 3.11.3!!!
import os

app = Flask('image_optimizer')


@app.route('/')
def show_index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def results():
    print(request.files)
    if request.method == 'POST':
        # write your function that loads the model
        # model = get_model()  # you can use pickle to load the trained model
        best_images = optimizer.run(request.files)
        # year = request.form['year']
        # pred = model.predict()
        # image_scores = run_model(model, request.files)
        return render_template('result.html', pred=123)


app.run("localhost", 9999, debug=True)
