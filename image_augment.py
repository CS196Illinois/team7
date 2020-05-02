
import cv2
import numpy as np

class ImageAugment():
    def __init__(self, image_rgb):
        image_rgb = np.array(image_rgb)
        self.image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        self.image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        self.image_rgb = image_rgb

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
        s=s+(255-s)*alpha
        s=np.clip(s, 0, 255)
        imghsv = cv2.merge([h,s,v])
        imghsv = cv2.cvtColor(imghsv.astype('uint8'), cv2.COLOR_HSV2RGB)
        return imghsv

    def adjust_sharpness(self, alpha):
        # alpha is the amount of blurriness you add. MUST be negative to sharpen. 0 does nothing, positive to blur
        blur = cv2.blur(self.image_rgb, (5, 5))
        return cv2.addWeighted(self.image_rgb, 1, blur, alpha, 5)


    def adjust_warmth(self, alpha, cool=False):
        (r, g, b) = cv2.split(self.image_rgb)
        increaseLUT = np.array([i + (255 - i) * alpha for i in np.arange(0, 256)])
        decreaseLUT = np.array([i * alpha for i in np.arange(0, 256)])
        if not cool:
            r = cv2.LUT(r, decreaseLUT)
            b = cv2.LUT(b, increaseLUT)
        else:
            r = cv2.LUT(r, increaseLUT)
            b = cv2.LUT(b, decreaseLUT)
        image_new = np.zeros_like(self.image_rgb)
        image_new[:,:,0] = r
        image_new[:, :, 1] = g
        image_new[:, :, 2] = b

        return image_new

    def adjust_cool(self, alpha, cool=True):
        self.adjust_warmth(alpha, cool)

    def augment(self):
        images = []
        images.append(self.adjust_brightness(.7))
        images.append(self.adjust_brightness(2))
        images.append(self.adjust_contrast(alpha=2, beta=3))
        images.append(self.adjust_saturation(.3))
        images.append(self.adjust_saturation(-.3))
        images.append(self.adjust_sharpness(.4))
        images.append(self.adjust_sharpness(-.4))
        images.append(self.adjust_warmth(.0001))
        images.append(self.adjust_cool(.000001))
        return images
        
        
