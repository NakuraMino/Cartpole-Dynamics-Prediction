from torch.utils.data import Dataset
import pandas as pd
import cv2
import os


class CartpoleDataset(Dataset):
    """
    Cartpole dataset class
    """

    def __init__(self, csv_file, path):
        """
        :arg csv_file: filename. string ('data.csv')
        :arg path: path to the csv file. string ('../data/image_dataset/')
                   (absolute path)
        """
        self.datafile = csv_file
        self.path = path
        self.data = pd.read_csv(self.path + self.datafile, header=None)

    def __len__(self):
        """
        Enables using len() on the object

        returns: length of dataset (# rows in csv)
        """
        return len(self.data)

    def get_composite_image(self, i, n, grayscale=False):
        """
        Fetches a composite image created by stitching n images
        with i being the index of the last-second element

        :arg i: index of the last-second image
        :arg n: the number of images to be fetched
        :arg grayscale: whether grayscale composite image is required

        returns: a composite image by stitching n images side-by-side
        """
        N = len(self.data)
        imdir, _ = os.path.split(self.path[:-1])
        imdir += '/'

        # Check if provided index is sensible
        if(i >= n-2 and i <= N-2 and n >=2):
            images = []  # will hold the images
            for j in range(i-n+1, i+2):
                im_file = imdir + self.data.iloc[j, 4]
                if(grayscale):
                    images.append(cv2.imread(im_file, cv2.IMREAD_GRAYSCALE))
                else:
                    images.append(cv2.imread(im_file, cv2.IMREAD_COLOR))
            composite_image = cv2.hconcat(images)
            return composite_image
        else:
            print('Index should be between {} - {} (provided {}) and the number of images should be greater than or equal to 2 (provided {}).'.format(n-2, N-2, i, n))


# This script is to be used just to load the class.
# The below code is just to test if it is working
# (with random images from the dataset)
if __name__ == '__main__':
    path = '/media/nishant/MyDrive/Acads/UW/2019-20/3Spring/CSE571-AI-BasedMobileRobotics/projects/project1/CSE571_Project1/data/image_dataset/'
    dataset = CartpoleDataset('data.csv', path)
    cv2.imshow('Composite image', dataset.get_composite_image(2, 4))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
