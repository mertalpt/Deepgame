# Deepfake creations abstracted

# Import from first-order-model repository
import demo as fomm_demo

import imageio
import numpy as np
import matplotlib.pyplot as plt

from DGImage import DGImage
from DGVideo import DGVideo


class ImageAnimater:
  """
    Provides a nice interface for quick and easy 'First Order Model' usage.
  """
  
  def __init__(self, config_path, checkpoint_path):
    """
      Initializes the ImageAnimater object with a pretrained model and
      its corresponding config file.
    """
    generator, kp_detector = fomm_demo.load_checkpoints(
        config_path=config_path, checkpoint_path=checkpoint_path)
    # Store into object
    self.generator = generator
    self.keypoint_detector = kp_detector


  def animate(self, driver, target, size=(256, 256)):
    """
      Uses First Order Model project to animate the target image using the
      driver video.

      :param driver: DGVideo object for the driver video.
      :param target: DGImage object for the target image.
      :param size: Tuple for dimensions to work with.
    """
    # An awkward way of resizing the inputs
    image = target.resize(size=size)
    video = driver.get_video()
    video = DGVideo.from_video(video, size=size, fps=driver.fps)
    video = driver.get_video(mode='rgb', type='float')
    image = image.get_image(mode='rgb', type='float')
    anim = fomm_demo.make_animation(image, video,
                               self.generator, self.keypoint_detector, 
                               relative=True)

    return DGVideo.from_video(anim, fps=driver.fps)
