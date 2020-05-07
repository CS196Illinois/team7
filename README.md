# Image Optimizer

**Authors:** Aileen Long, Ishita Rao, Reece Anderson

---

An image optimizer targeted at maximising the number of likes you could get on social media.

**Note:** Before running the application, be sure to change UPLOAD_FOLDER in app.py. It depends on your OS, but it should be an absolute path to team7/app/uploads. You can obtain this path by right-clicking the uploads folder and selecting 'Get Info' or 'Properties'.

The Image Optimizer allows the user to load a set of images into our model, which will then generate augmented versions of the original images. The best of these augmented images will be returned to the user to share with friends and family.

### Dependencies
* The trained models (see below). They were trained on thousands of popular Instagram images.
* [Flask](https://pypi.org/project/Flask/)
* [Tensorflow 2.1.0](https://pypi.org/project/tensorflow/)
* [NumPy](https://pypi.org/project/numpy/)
* [Pillow 7.1.2](https://pypi.org/project/Pillow/)
* [opencv-python 4.2.0.34](https://pypi.org/project/opencv-python/)

### Models
* unpruned model: https://drive.google.com/file/d/19u1gfC5yMD3xnbk7yBuLEAmu9bfkB4zq/view?usp=sharing
* pruned model: https://drive.google.com/file/d/1-FO5C8Pi-RDbYNlAEpTIiLdz9jwSYP1b/view?usp=sharing

Once you run the application, it will be hosted on http://localhost:9999/, which you can access from your browser. You may upload as many images as you want. The model takes time to run, so you will need to wait for a short period of time to get the results. 

We believe that the Image Optimizer will help ensure people can stay in the moment and spend less time worrying about selecting and editing the picture to be the best it can be. 
