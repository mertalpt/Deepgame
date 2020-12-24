# Style Transfer abstracted
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub

from DGImage import DGImage


class StyleTransferer:
  """
    Provides a nice interface for quick and easy 'Style Transfer' usage.
  """

  def __init__(self, module=None):
    """
      Simple wrapper for TF-Hub module.
    """
    # Import ready made ML model
    if module is None:
      print('Loading module from TF Hub. This may take 5-10 minutes.')
      self.hub_module = tf_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    else:
      self.hub_module = module


  def stylize(self, style_image, content_image):
    """
      Produces stylized result of the content image.

      :param style_image: DGImage to use as the style.
      :param content_image: DGImage to use as the content.
      :returns: DGImage of the content styled with the style image.
    """
    # ML model input format requires float32 data
    content = content_image.get_image(mode='rgb', type='float').astype(np.float32)[np.newaxis, ...]
    style = style_image.get_image(mode='rgb', type='float').astype(np.float32)[np.newaxis, ...]
    style_NN = tf.nn.avg_pool(style, ksize=[3,3], strides=[1,1], padding='SAME')
    outputs = self.hub_module(tf.constant(content), tf.constant(style_NN))
    
    # 0 is where the stylized image lives
    tmp = outputs[0][0].numpy()

    return DGImage.from_image(tmp)
