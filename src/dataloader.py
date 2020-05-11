from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import cv2
import os


class CartpoleDataset(Dataset):
    """
    Cartpole dataset class
    """

    def __init__(self, csv_file, path, num_images, H=128, W=128, grayscale=True):
        """
        :arg csv_file: filename. string ('data.csv')
        :arg path: path to the csv file. string ('../data/image_dataset/')
                   (absolute path)
        :arg num_images: # of images to be returned by get_ith_image_set()
        :arg W: width of a single image
        :arg H: height of a single image
        :arg grayscale: whether grayscale images are to be stored in the dataset
        """
        self.datafile = csv_file
        self.path = path
        self.data = pd.read_csv(self.path + self.datafile, header=None)
        self.n = num_images
        self.W = W
        self.H = H
        self.grayscale = grayscale

    def __len__(self):
        """
        Enables using len() on the object

        returns: length of dataset (# rows in csv)
        """
        return len(self.data)

    def __getitem__(self, i):
        """
        Makes the dataset object iterable.

        :arg i: index of the last-second image

        returns: a tuple (images, delta_states)
        - images is a 4d torch tensor carrying n images:
            for grayscale - (1 x n x H x W)
            for color - (3 x n x H x W)
          with i being the index of the last-second image.
        - delta_states is a 2d numpy array (n x 4)
          containing the 4 delta states for the n images
        """
        N = len(self.data)
        imdir, _ = os.path.split(self.path[:-1])
        imdir += '/'

        all_delta_states = self.data.iloc[:, 0:4].to_numpy()

        # Check if provided index is sensible
        if(i >= self.n-2 and i <= N-2 and self.n >= 2):
            delta_states = np.zeros((self.n, 4))
            if(self.grayscale):
                images = torch.empty(self.n, 1, self.H, self.W, dtype=torch.uint8)
            else:
                images = torch.empty(self.n, 3, self.H, self.W, dtype=torch.uint8)
            k = 0
            for j in range(i-self.n+2, i+2):
                im_file = imdir + self.data.iloc[j, 4]
                if(self.grayscale):
                    images[k] = torch.from_numpy(cv2.imread(im_file, 0).reshape((1, self.H, self.W)))
                else:
                    images[k] = torch.from_numpy(np.swapaxes(cv2.imread(im_file, 1), 0, 2))
                delta_states[k] = all_delta_states[j]
                k += 1
            images = np.swapaxes(images, 0, 1)
            return (images, delta_states)
        else:
            print('Index should be between {} - {} (provided {}) and the number of images should be greater than or equal to 2 (provided {}).'.format(self.n-2, N-2, i, self.n))


# This script is to be used just to load the class.
# The below code is just to test if it is working
# (with random images from the dataset)
if __name__ == '__main__':
    path = '/media/nishant/MyDrive/Acads/UW/2019-20/3Spring/CSE571-AI-BasedMobileRobotics/projects/project1/CSE571_Project1/data/image_dataset/'
    dataset = CartpoleDataset('data.csv', path, 5)
    images = dataset[4][0].numpy()
    print(images.shape)
