import random
import numpy as np
import torch
from torch.autograd import Variable


class ImagePool():
    """
    This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        #pool_size is the size of image buffer , if pool_size = 0, no buffer will be created
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return Variable(images)
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size: #If the buffer is not full, keep inserting images into it
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:     #50% chance, buffer will return a previously stored image and insert the current image into buffer
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image) #Another 50% chance, will return the current image 
        return_images = Variable(torch.cat(return_images, 0)) # collect all images and return
return return_images
