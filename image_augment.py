
import cv2
import numpy as np

class ImageAugment():
    def __init__(self, image_rgb):
        self.image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        self.image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        self.image_rgb = image_rgb



    def adjust_brightness(self, alpha):
        assert alpha<0 and alpha>1
        brighter_lab = np.add(self.image_lab[:, :, 0], (1 - self.image_lab[:, :, 0])*alpha)
        return cv2.cvtColor(brighter_lab, cv2.COLOR_LAB2RGB)

    def adjust_brightness_gamma(self, gamma):
        LUT  = np.array([((i / 255.0) ** gamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(self.image_rgb, LUT)

    def adjust_contrast(self, alpha=2, beta=8):
        clahe = cv2.createCLAHE(clipLimit=alpha, tileGridSize=(beta,beta))
        return clahe.apply(self.image_rgb)

    def adjust_saturation(self, alpha):
        temp_hsv = self.image_hsv.astype('float32')
        (h, s, v) = cv2.split(temp_hsv)
        s=s*alpha
        s=np.clip(s, 0, 255)
        imghsv = cv2.merge([h,s,v])
        imghsv = imghsv.astpye('uint8')
        return imghsv

    def adjust_sharpness(self, alpha, beta=5):
        # alpha is the amount of blurriness you add. MUST be negative to sharpen. 0 does nothing, positive to blur
        blur = cv2.GaussianBlur(self.image_rgb, (1, 1), beta)
        return cv2.addWeighted(self.image_rgb, 1, blur, alpha)

    def adjust_blur(self, alpha, beta=5):
        return self.adjust_sharpness(alpha, beta)

    def adjust_warmth(self, alpha, cool=False):
        (r, g, b) = cv2.split(self.image_rgb)
        increaseLUT = np.array([i + (255 - i) * alpha for i in np.arange(0, 256)])
        decreaseLUT = np.array([i * alpha for i in np.arange(0, 256)])
        if cool:
            r = cv2.LUT(r, decreaseLUT)
            b = cv2.LUT(b, increaseLUT)
        else:
            r = cv2.LUT(r, increaseLUT)
            b = cv2.LUT(b, decreaseLUT)
        return cv2.merge([r,g,b])

    def adjust_cool(self, alpha, cool=True):
        self.adjust_warmth(alpha, cool)

