from PIL import Image
import tensorflow as tf


# calls the other functions to return the dict containing the best images
def run(images):
    processed_images = run_preprocessor(images)
    model = get_model()
    best_images = run_model(model, processed_images)
    return best_images


def get_model():
    model = tf.keras.models.load_model('/Users/ishita/team7/app/model_ver2.h5')
    return model


# returns a dict containing the image and its score
def run_model(model, files):
    augmented_images = model.augment(files)
    scores = model.calculate_socres(augmented_images)  # list or a dict
    best_images = {}
    for index in range(scores.len):
        best_images.update({augmented_images.get(index): scores[index]})

    return best_images


# resizes images and does whatever else a preprocessor is supposed to do
def run_preprocessor(images):
    processed_images = []

    for img_path in images:
        image = Image.open(img_path)
        image = image.resize((200, 200))
        processed_images.append(image)

    return processed_images
