import skimage as sk
import skimage.io as sk_io
import skimage.transform as sk_transform
import skimage.color as sk_color

from matplotlib import pyplot as plt
from autocrop import Cropper

import cv2
import numpy as np
import tensorflow as tf

### Class Definition

class DGImage:
  """
    Unified image object for DeepGame usage.
    Forces images to a 8-bit RGBA format.
  """

  ### Constructor

  def __init__(self, data, size=None):
    """
      Takes the image data as a Numpy ndarray.
      Do not call directly, use static constructors below instead.

      :param data: Image data as a Numpy ndarray
      :param size: Tuple of image (width, height)
      :returns: DGImage object containing a formatted image
    """
    # Sanity check
    if len(data.shape) != 3 or data.shape[2] < 3:
      raise ValueError(
          'Image shape {} is unexpected for a color image.'.format(str(data.shape)))

    # Copy first
    tmp = data.copy()

    # Resize if requested
    if size is not None:
      tmp = sk_transform.resize(tmp, size, anti_aliasing=True)
    else:
      size = data.shape[:2]

    # Force 8-bit depth
    tmp = sk.img_as_ubyte(tmp)

    # Ensure existance of alpha channel
    if tmp.shape[2] == 3:
      alpha = np.full(shape=size, fill_value=255, dtype=np.dtype(np.uint8))
      tmp = np.dstack((tmp, alpha))

    # Done
    self.image = tmp


  ### Builders

  def from_path(path, size=None):
    """
      Reads the image from a path and formats it into a DGImage.
      Pass a tuple to 'size' if resizing of the image is desired.

      :param path: String pointing to the path of the image.
      :param size: Tuple for image dimensions for the formatting.
      :returns: DGImage with the formatted image inside.
    """
    data = sk_io.imread(path)
    return DGImage(data, size)

  
  def from_image(data, size=None):
    """
      Takes the image as a numpy array and formats it into a DGImage.
      Pass a tuple to 'size' if resizing of the image is desired.

      :param data: Numpy array holding the image data.
      :param size: Tuple for image dimensions for the formatting.
      :returns: DGImage with the formatted image inside.
    """
    return DGImage(data, size)


  ### Public Methods

  def save(self, path):
    """
      Saves the image into the given path.

      :param path: String of a path to save the file in.
    """
    sk_io.imsave(path, self.image)


  def shape(self):
    """
      Gets the image shape.

      :returns: Dimension of the inner image.
    """
    return self.image.shape


  def resize(self, size):
    """
      Copies the image and resizes it.

      :param size: New dimensions.
      :returns: Resized copy of the original image.
    """
    return DGImage.from_image(self.image, size=size)

  
  def get_image(self, mode='rgba', type='int'):
    """
      Returns a copy of the image data with the given mode.
      Possible modes: 'rgba', 'rgb', 'gray'.
      Possible types: 'int' (-> uint8), 'float' (-> float64).

      :param mode: Color mode to apply.
      :param type: Data type of the result.
      :returns: Numpy ndarray for the image.
    """
    if type not in ['int', 'float']:
      raise ValueError('Unrecognized value {} for type.'.format(str(type)))

    # Get data
    if mode == 'rgba':
      tmp = self.image.copy()
    elif mode == 'rgb':
      tmp = self._image_in_rgb()
    elif mode == 'gray':
      tmp = self._image_in_gray()
    else:
      raise ValueError('Unrecognized value {} for mode.'.format(str(mode)))
    
    # Cast and normalize
    if type == 'int' and tmp.dtype != np.uint8:
      tmp = tmp * 255.0
      tmp = tmp.astype(np.uint8)
    elif type == 'float' and tmp.dtype == np.uint8:
      tmp = tmp.astype(np.float64)
      tmp = tmp / 255.0

    # Sanity check
    if type == 'int' and tmp.dtype != np.uint8:
      raise Error('Type casting failed.')
    elif type =='float' and tmp.dtype != np.float64:
      raise Error('Type casting failed.')

    return tmp


  def extract_face(self, size=None, face_percent=50, padding=None, fix_gamma=True):
    """
      Uses face cropper of 'autocrop' pip package to extract the face 
      from the image. Check the documentation linked below for more information.

      https://github.com/leblancfg/autocrop

      :param size: Tuple for dimensions of the result.
      :param face_percent: How much of the image should consist of the face.
      :param padding: Space to include beyond the face bounding box.
      :param fix_gamma: Fixes exposure issues resulting from cropping.
      :returns: DGImage containing the extraction result.
    """
    # If no special size, use image's own size
    if size is None:
      size = self.shape()[:2]

    # Manipulate image from RGB to BGR format
    # Because autocrop assumes BGR input
    image = cv2.cvtColor(self.get_image(mode='rgb'), cv2.COLOR_RGB2BGR)

    # Initialize cropper
    cropper = Cropper(
        width=size[1], height=size[0], 
        face_percent=face_percent, padding=padding, fix_gamma=fix_gamma)
    cropped = cropper.crop(image)

    # If no face was found, will return None
    if cropped is None:
      return None

    return DGImage.from_image(cropped)


  def filter_luminance(self, lower=0.0, upper=1.0):
    """
      Returns a list of points filtered by their luminance.

      :param lower: Lower limit for luminance in 0-1 floating point range.
      :param upper: Upper limit for luminance in 0-1 floating point range.
      :returns: Numpy array of points falling into given luminance limits.
    """
    image = self.get_image(mode='gray')
    brighter = np.array(lower < image)
    darker = np.array(image < upper)
    points = np.logical_and(brighter, darker)
    points = np.array((points).nonzero()).T
    return points


  def crop_center(self):
    """
      Turns the image dimensions to a square by taking the center square and
      clipping the remainder.

      :returns: DGImage copy cropped and centered from the original.
    """
    image = self.image.copy()
    shape = image.shape
    new_shape = min(shape[0], shape[1])
    dx = (shape[0] - new_shape) // 2
    dy = (shape[1] - new_shape) // 2
    image = image[dx:new_shape+dx, dy:new_shape+dy]
    return DGImage.from_image(image)


  ### Private Methods

  def _image_in_gray(self):
    """
      Returns image data in grayscale.

      :returns: Numpy float64 array for image in grayscale.
    """
    image = self.image.copy()
    return sk_color.rgb2gray(image)


  def _image_in_rgb(self):
    """
      Returns image data in RGB instead of RGBA.

      :returns: Numpy array with three color channels.
    """
    image = self.image.copy()
    return image[..., :3]


  ### Static Methods

  def display(*images, titles=None, ncols=4, dpi=200.0, op=None):
    """
      Displays one or more DGImages using matplotlib.

      :param images: List of DGImage objects.
      :param titles: List of strings to title each image with, if desired.
      :param ncols: Number of columns to display the images in.
      :param dpi: Pixels per inch to display.
      :param op: Operation to apply to each image as they are displayed.
    """
    if titles is not None and len(images) != len(titles):
      raise ValueError('Number of images do not match the number of titles.')

    # Set default operation, extract image unchanged
    if op is None:
      op = lambda x: x.image

    # In case division rounds down to 0, bump to 1
    nrows = (len(images) // ncols) + 1

    # Construct the plot
    fig, axes = plt.subplots(nrows, ncols, dpi=dpi)
    fig.tight_layout()
    axes = axes.ravel()
    index = 0

    for image in images:
      tmp = op(image)
      curr = axes[index]

      if titles is None:
        curr.set_title('Index: ' + str(index))
      else:
        curr.set_title(titles[index])

      curr.imshow(tmp)
      index += 1

    # Cover unused axes
    while index < nrows * ncols:
      axes[index].set_visible(False)
      index += 1

    # Done
    plt.show()
