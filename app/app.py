from flask import Flask, render_template, request

app = Flask('image_optimizer')

from types import SimpleNamespace
def dummy():
    return 123
def get_model():
    model = SimpleNamespace(predict=dummy)
    return model

@app.route('/')
def show_index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def results():
    print(request.files)
    if request.method == 'POST':
        #write your function that loads the model
        model = get_model() #you can use pickle to load the trained model
        # year = request.form['year']
        pred = model.predict()
        return render_template('result.html')

app.run("localhost", "9999", debug=True)