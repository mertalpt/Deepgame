from matplotlib import pyplot as plt
from matplotlib import animation

import skimage as sk
import skimage.io as sk_io
import skimage.transform as sk_transform
import skimage.color as sk_color

import os
import imageio
import numpy as np

from DGImage import DGImage


### Class Definition

class DGVideo:
  """
    Unified video object for DeepGame usage.
    Holds a list of frames that can be converted to DGImage objects on demand.
  """

  ### Constructor

  def __init__(self, data, size, fps):
    """
      Constructs a DGVideo from the given list of numpy arrays.
      Intended to be called by the builders. Do not use directly.

      :param data: List of numpy arrays corresponding to a frame.
      :param size: Tuple for the dimensions of the frames.
      :param fps: FPS of the video.
    """
    if len(data) == 0:
      raise ValueError('DGVideo constructor was passed empty data.')

    self.fps = fps
    self.frames = list()

    for frame in data:
      dg_frame = DGImage.from_image(frame, size)
      self.frames.append(dg_frame)


  ### "Private" Methods

  def __len__(self):
    return len(self.frames)


  ### Builders

  def from_path(path, size=None, fps=None):
    """
      Builds a DGVideo from the given video file.

      :param path: String to path of the video file.
      :param size: Tuple for the dimensions of the frames.
      :param fps: FPS override of the video file.
      :returns: DGVideo object.
    """

    reader = imageio.get_reader(path)
    if fps is None:
      fps = reader.get_meta_data()['fps']
    frames = list()
    for frame in reader:
      frames.append(frame)
    reader.close()

    video = DGVideo(frames, size, fps)
    return video


  def from_video(data, size=None, fps=30):
    """
      Builds a DGVideo from the given video as data.

      :param data: List of numpy arrays corresponding to frames.
      :param size: Tuple for the dimensions of the frames.
      :param fps: FPS override of the video file.
      :returns: DGVideo object.
    """
    return DGVideo(data, size, fps)


  ### Methods

  def save(self, path, mode='rgba', type='int'):
    """
      Saves the video to the given path.

      Possible modes: 'rgba', 'rgb', 'gray'.
      Possible types: 'int', 'float'.

      :param mode: Color mode to use for the frames.
      :param type: Data type for the frames.
      :param path: Path to save the video to.
    """
    formatted_frames = list()

    for frame in self.frames:
      formatted_frames.append(frame.get_image(mode=mode, type=type))

    imageio.mimsave(path, formatted_frames, fps=self.fps)


  def save_to_directory(self, path, mode='rgba', type='int', prefix='frame-'):
    """
      Saves the video as a series of images inside the given directory.
      Creates the directory if it does not exist.

      :param path: Directory path.
      :param mode: Color mode to use for the frames.
      :param type: Data type for the frames.
      :param prefix: Prefix to use for the naming of the images.
    """
    if not isinstance(path, str):
      raise ValueError('Given directory path {} is invalid.'.format(str(path)))
    if not isinstance(prefix, str):
      raise ValueError('Given prefix {} is not a valid string.'.format(str(prefix)))
    if os.path.isfile(path):
      raise ValueError('Given directory path {} exists as a file.'.format(str(path)))

    if not os.path.exists(path):
      os.mkdir(path)

    for i, frame in enumerate(self.frames):
      tmp = '{}/{}{}.png'.format(path, prefix, str(i))
      frame.save(tmp)


  def get_frame(self, index):
    """
      Gets the frame with the given index as a DGImage object.

      :param index: Index of the frame in the video.
      :returns: The frame as a DGImage object.
    """

    if index >= len(self):
      raise IndexError('Given index {} is out of bounds for {} frames.'.format(
              str(index), str(len(self))))

    return self.frames[index]


  def get_video(self, mode='rgba', type='int'):
    """
      Gets the video data in the given format.

      Possible modes: 'rgba', 'rgb', 'gray'.
      Possible types: 'int', 'float'.

      :param mode: Color mode to use for the frames.
      :param type: Data type for the frames.
      :param path: List of Numpy arrays corresponding to the frames.
    """
    frames = list()

    for frame in self.frames:
      tmp = frame.get_image(mode=mode, type=type)
      frames.append(tmp)

    return frames


  def extract_faces(self, size=None, face_percent=50, padding=None, fix_gamma=True):
    """
      Uses face cropper of 'autocrop' pip package to extract the face
      from the image. Check the documentation linked below for more information.

      https://github.com/leblancfg/autocrop

      :param size: Tuple for dimensions of the result.
      :param face_percent: How much of the image should consist of the face.
      :param padding: Space to include beyond the face bounding box.
      :param fix_gamma: Fixes exposure issues resulting from cropping.
      :returns: DGVideo containing frames with extracted faces.
    """
    frames = list()

    for frame in self.frames:
      tmp = frame.extract_face(size, face_percent, padding, fix_gamma)
      # If no faces are found, it will return None
      if tmp is not None:
        frames.append(tmp.get_image())

    return DGVideo.from_video(frames, fps=self.fps)


  def swap_color_range(self, lower, upper, new_color):
    """
      Swaps the color of every pixel that falls within the color range to the
      given new color. Does this for every frame in the video and produces
      a new video.

      :param lower: 8-bit RGBA color lower limit of the pixels to change.
      :param upper: 8-bit RGBA color upper limit of the pixels to change.
      :param new_color: 8-bit RGBA color to swap with the old color.
      :returns: New DGVideo with colors swapped for every frame.
    """
    tmp = list()
    for frame in self.frames:
      curr = frame.swap_color_range(lower, upper, new_color)
      tmp.append(curr.image)
    return DGVideo.from_video(tmp)


  def animation(self, title='Animated Video', interval=None, repeat_delay=1000):
    """
      Returns the video as a matplotlib ArtistAnimation.

      :param title: Title of the animated video.
      :param interval: Milliseconds between frames.
      :returns: ArtistAnimation representation of the video.
    """
    if interval is None:
      interval = max(int(1000 / self.fps), 1)

    figure = plt.figure()
    plt.title(title)

    frames = self.get_video()
    converted = list()

    for frame in frames:
      tmp = plt.imshow(frame, animated=True)
      converted.append([tmp])

    anim = animation.ArtistAnimation(figure, converted,
                                     interval=interval,
                                     repeat_delay=repeat_delay)
    plt.close()

    return anim
