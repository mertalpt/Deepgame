import matplotlib.pyplot as plt

from skimage import filters, color, morphology
from scipy import ndimage as ndi

import numpy as np

from DGImage import DGImage


class MaskOperations:

  def apply(raw_image, mask, mask_color=[0, 255, 0, 255]):
    """
      Applies the given mask to the given image and returns the result.
    """
    masked = raw_image.get_image()
    masked[mask] = np.array(mask_color)
    return DGImage.from_image(masked)


  def generate(raw_image, light_mod=0.98, dark_mod=0.02, sigma=2.0, seed=None, size_lim=0):
    """
      Generates possible mask pieces of an image by segmenting the image
      using bright and dark spots in the image. Watershed transformation
      is used to grow the spots into segments, which then are output as masks.

      Alternatively, a list of (Bool, Int, Int) tuples can be passed in 'seed'
      to manually segment the picture. Integers denote the pixel position and
      the boolean denotes whether the segment around the pixel should be
      included in the resulting mask.
    """
    # Change image to monochrome for easier handling
    image = raw_image.get_image(mode='gray', type='float')

    # Apply filtering to get contours
    sobel = filters.sobel(image)
    blurred = filters.gaussian(sobel, sigma=sigma)

    if seed is None:
      # Find points on the image based on brightness
      light_spots = np.array((image > light_mod).nonzero()).T
      dark_spots = np.array((image < dark_mod).nonzero()).T
      # Create the section mask
      bool_mask = np.zeros(image.shape, dtype=np.bool)
      bool_mask[tuple(light_spots.T)] = True
      bool_mask[tuple(dark_spots.T)] = True
      seed_mask, num_seeds = ndi.label(bool_mask)
    else:
      seed_mask = np.zeros(image.shape, dtype=np.int)
      for keep, i, j in seed:
        # As long as they are separate, exact values do not matter
        seed_mask[i][j] = 2 if keep else 1

    # Generate watershed from seed mask
    ws = morphology.watershed(blurred, seed_mask)

    # Generate masks
    masks = list()
    for section in set(ws.ravel()):
      if np.sum(ws == section) > size_lim:
        mask = (ws == section)
        masks.append(mask)

    return masks


  def display(image, masks, mask_color=[0, 255, 0, 255]):
    """
      Applies masks from the given list to the given image and displays
      each using matplotlib.pylot . 
      
      Intended use is to assist with construction of a mask that will leave
      only the face in the image by displaying the application of each given
      mask. Then, using the set operations on masks a final mask can be
      constructed from the pieces. 
    """
    image = image.get_image(type='float')

    ncols = 4
    nrows = (len(masks) // ncols) + 1

    fig, axes = plt.subplots(nrows, ncols, dpi=200.0)
    fig.tight_layout()
    axes = axes.ravel()
    index = 0

    for mask in masks:
      tmp = image.copy()
      tmp[mask] = np.array(mask_color)
      curr = axes[index]
      curr.set_title('Mask Index: ' + str(index))
      curr.imshow(tmp)
      index += 1

    # We will have unused axes
    while index < nrows * ncols:
      axes[index].set_visible(False)
      index += 1

    plt.show()


  ### Set operations
  def union(*masks):
    acc_mask = np.zeros(masks[0].shape, dtype=bool)
    for mask in masks:
        acc_mask = np.logical_or(acc_mask, mask)

    return acc_mask


  def intersection(*masks):
    acc_mask = np.ones(masks[0].shape, dtype=bool)
    for mask in masks:
        acc_mask = np.logical_and(acc_mask, mask)

    return acc_mask


  def difference(mask_1, mask_2):
    intersection = MaskOperations.intersection(mask_1, mask_2)
    res = mask_1.copy()
    res[intersection] = False
    return res
