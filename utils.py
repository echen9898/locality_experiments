
import os
import numpy as np

def create_path(path):
    """ Checks if a path exists. If it doesn't, create
    the path recursively
    """
    if not os.path.isdir(path):
        os.makedirs(path)

def log_example_img(dataset, experiment, num_channels=1, train=True):
    
    for index in range(10):
        tmp, _ = dataset[index]

        if num_channels == 1: # grayscale
            image_channels = 'last'
            img = tmp.numpy()[0]
        elif num_channels == 3: # RGB
            image_channels = 'first'
            img = tmp.numpy()

        if train:
            data = experiment.log_image(img, image_channels=image_channels, name="train_%d.png" % index)
        else:
            data = experiment.log_image(img, image_channels=image_channels, name="test_%d.png" % index)

class ScrambleImg(object):
    """Scramble images (both grayscale and RGB).
    Fixed scramble - each image is scrambled deterministically.
    Random scramble - each image is randomly scrambled.
    """

    def __init__(self, fixed_scramble=True, dim=28):
        self.fixed_scramble = fixed_scramble
        self.dim = dim
        self.scramble_indx = np.arange(self.dim*self.dim)
        np.random.shuffle(self.scramble_indx)

    def __call__(self, pic):
        img = np.array(pic)
        if len(img.shape) == 2: # greyscale
            h, w = img.shape
            scramble_indx = self.scramble_indx.copy()
            if self.fixed_scramble == False:
                np.random.shuffle(scramble_indx)
            img = img.reshape(1, h*w)
            img = img[0, scramble_indx]
            img = img.reshape(h, w, 1)
        elif len(img.shape) == 3: # RGB
            h, w, _ = img.shape
            scramble_indx = self.scramble_indx.copy()
            if self.fixed_scramble == False:
                np.random.shuffle(scramble_indx)
            scrambled = np.empty((h, w, 3))
            r_layer = img[:,:,0].reshape(1, h*w)[0, scramble_indx]
            g_layer = img[:,:,1].reshape(1, h*w)[0, scramble_indx]
            b_layer = img[:,:,2].reshape(1, h*w)[0, scramble_indx]
            img = np.empty((h, w, 3))
            img[:,:,0] = r_layer.reshape(h, w)
            img[:,:,1] = g_layer.reshape(h, w)
            img[:,:,2] = b_layer.reshape(h, w)

        return img.astype('uint8')











