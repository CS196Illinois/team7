# Image Optimizer
An image optimizer targeted at maximising the number of likes you could get on social media.

The Image Optimizer allows the user to load a set of images into our model, which will then generate augmented versions of the original images. The best of these augmented images will be returned to the user to share with friends and family.

### Dependencies
* The trained [model](https://drive.google.com/file/d/1-2wrpw22OGTGf5q3TIDZ_gN_n8GQTBDY/view?usp=sharing). It was trained on thousands of popular Instagram images.
* [Flask](https://pypi.org/project/Flask/)
* [Tensorflow 2.1.0](https://pypi.org/project/tensorflow/)
* [NumPy](https://pypi.org/project/numpy/)
* [Pillow 7.1.2](https://pypi.org/project/Pillow/)
* [opencv-python 4.2.0.34](https://pypi.org/project/opencv-python/)

Once you run the application, it will be hosted on http://localhost:9999/, which you can access from your browser. You may upload as many images as you want. The model takes time to run, so you will need to wait for a short period of time to get the results. 

We believe that the Image Optimizer will help ensure people can stay in the moment and spend less time worrying about selecting and editing the picture to be the best it can be. 
