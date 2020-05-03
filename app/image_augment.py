import ntpath
import cv2
import numpy as np
from PIL import Image

class ImageAugment():
    def __init__(self, image_rgb, filename):
        image_rgb = np.array(image_rgb)
        self.image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        self.image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        self.image_rgb = image_rgb
        self.filename = filename
        self.paths = []

    def adjust_brightness(self, gamma):
        # <1 means brighter, >1 means darker
        # For reference you probably want the range [.5, 5]
        LUT  = np.array([((i / 255.0) ** gamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(self.image_rgb, LUT)

    def adjust_contrast(self, alpha=3, beta=6):
        lab = self.image_lab
        clahe = cv2.createCLAHE(clipLimit=alpha, tileGridSize=(beta, beta))
        l = lab[:,:,0]
        lab[:,:,0] = clahe.apply(l)
        RGB = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return RGB

    def adjust_saturation(self, alpha):
        temp_hsv = self.image_hsv.astype('float32')
        (h, s, v) = cv2.split(temp_hsv)
        s=s+(255-s)*alpha**3
        s = np.clip(s, 1, 254)
        imghsv = cv2.merge([h,s,v])
        imghsv = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2RGB)
        return imghsv

    def adjust_sharpness(self, alpha):
        # alpha is the amount of blurriness you add. MUST be negative to sharpen. 0 does nothing, positive to blur
        blur = cv2.blur(self.image_rgb, (5, 5))
        return cv2.addWeighted(self.image_rgb, 1, blur, alpha, 5)

    def adjust_warmth(self, alpha, cool=False):
        (r, g, b) = cv2.split(self.image_rgb)
        if not cool:
            r = r*(1-alpha**2)
            b = b+(255-b)*alpha**2
        else:
            r = r+(255-r)*alpha**2
            b = b*(1-alpha**2)
        image_new = np.zeros_like(self.image_rgb)
        image_new[:, :, 0] = r
        image_new[:, :, 1] = g
        image_new[:, :, 2] = b

        return image_new

    def adjust_cool(self, alpha, cool=True):
        return self.adjust_warmth(alpha, cool)

    def augment(self):
        images = [self.adjust_brightness(.8), self.adjust_brightness(1.7), self.adjust_contrast(alpha=2, beta=3),
                  self.adjust_saturation(.5), self.adjust_saturation(-.5), self.adjust_sharpness(.3),
                  self.adjust_sharpness(-.3), self.adjust_warmth(.4), self.adjust_cool(.4)]
        self.save_images(images)
        return images

    def save_images(self, images):
        for index in range(len(images)):
            pil = Image.fromarray(images[index])

            # basename gets the filename; splitext separates the extension from the filename
            filename = ntpath.splitext(ntpath.basename(self.filename))[0]
            path = "./uploads/" + filename + "_" + str(index) + ".png"
            self.paths.append(path)
            pil.save(path)

    def get_paths(self):
        return self.paths
        
        
        
        
