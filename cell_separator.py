"""
Cell separator for masks that are produced by deepflash
"""
import os

#from utils import *
import numpy as np
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import cv2
from skimage.feature import peak_local_max
from scipy import ndimage
from skimage.segmentation import watershed


Image.MAX_IMAGE_PIXELS = 400000000

def cell_sep(mask, min_distance=15, exclude_border=True, plot_result=False):
    """
    separate a mask of cells into separate cells. This code takes care of overlapping cells
    :param mask: 2d np array, with two values: 0 for background and 255 for forground (forground
                 can be any value, but background should be 0)
    :param exclude_border: whether to exclude the border when finding cell centers
    :param plot_result: whether to visualize the result of segmentation
    :return: mask-shaped array with different values for each cell
    """
    # make sure the mask only has two values: 0 and 255
    mask = (mask/mask.max() * 255).astype('uint8')
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Distance transform to estimate cell centers ...
    # This line calculates the distance between each forground pixel and the background
    distance_transform = ndimage.distance_transform_edt(binary_mask)

    # Finding local maxima to estimate cell centers.
    local_maxi = peak_local_max(distance_transform, min_distance=min_distance, labels=binary_mask,
                                exclude_border=exclude_border)
    max_mask = np.zeros(mask.shape, dtype=bool)
    max_mask[tuple(local_maxi.T)] = True

    # Marker labelling ... each peak point will be given a label from 1 to n (# of clusters)
    markers, _ = ndimage.label(max_mask)

    # Perform the watershed segmentation
    labels = watershed(-distance_transform, markers, mask=binary_mask)

    if plot_result:
        # Visualize the result
        plt.figure(figsize=(12, 6))

        # Original mask
        plt.subplot(1, 2, 1)
        plt.imshow(mask, cmap='gray')
        plt.title('Original Mask')

        # Splitted mask
        plt.subplot(1, 2, 2)
        plt.imshow(labels, cmap='nipy_spectral')
        plt.title('Splitted Mask')
        plt.scatter(local_maxi.T[1], local_maxi.T[0], color='blue', s=4)
        plt.show()

    return labels


root_dir = Path(r"/Volumes/Maria_3/MASTER THESIS_MARIA/Predictions_DS8/NF200_pred/masks")
new_dir = Path(r"/Volumes/Maria_3/MASTER THESIS_MARIA/RNA Scope Project/analysis/masks/NF200")

if not os.path.exists(new_dir):
    os.makedirs(new_dir)
for img_dir in root_dir.rglob('*.png'):
    print(f'Processing {img_dir.name}')
    newpath = new_dir/img_dir.name
    if not os.path.exists(newpath):

        img = np.array(Image.open(img_dir))
        new_img = cell_sep(img)
        new_img = new_img.astype('uint16') # It is unlikely to have more than 65535 cells
        Image.fromarray(new_img).save(newpath)

