# !/usr/bin/python
import os  # Operating systems operations: file, directory etc.
import config  # Import configuration variables from config.py
import numpy as np  # Numerical calculations
from numpy import random  # Import ´random´ from numpy
from keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm  # Progress bar


def load_paths(root_path=config.TRAIN_DIR):
  '''
  Load image paths in the MURA dataset.

  args:
  root_path -- MURA dataset's path.

  returns:
  Paths -- List object that holds paths of all image files.
  Labels -- Numpy array object that holds labels of all image files.
  '''

  print("[INFO] Loading image paths...\n")

  Paths = []  # List object for storing image paths
  Labels = []  # List object for storing image labels

  for root, dirs, files in tqdm(os.walk(root_path)):  # Read all image files via os.walk which returns and iterator generator that traverses all files
    for image_name in files:
      path = os.path.join(root, image_name)
      Paths.append(path)
      label = 1 if root.split('_')[-1] == 'positive' else 0   # Replace positive labels with one, and negative labels with zero
      Labels.append(label)

  print("[INFO] There are %d images in the dataset.\n" % len(Paths))
  Paths = np.asarray(Paths)  # Convert to Numpy array
  Labels = np.asarray(Labels)  # Convert to Numpy array

  return Paths, Labels


def load_images(image_paths, size=config.IMAGE_SIZE, segmentation=False, configuration=None):
  '''
  Load images in the MURA dataset from their paths.

  args:
  image_paths -- Paths of image files.
  size -- Size of the image.
  segmentation -- Set ´true´ in order to apply segmentation.
  configuration -- Set ´pretrained´ in order to use pretrained models.

  returns:
  Images -- Numpy array object that holds pixel values of all image files.
  '''

  print("[INFO] Loading images...\n")
  Images = []  # List object for storing image data
  num_samples_succesfull = 0  # Number of images that succesfully loaded 
  num_samples_failed = 0  # Number of images that failed to load

  for image_path in tqdm(image_paths):
    try:
      img = load_img(image_path, color_mode='grayscale', target_size=config.IMAGE_SIZE)  # Load the image as graysclae and resize the image to be 512 x 512
      img = random_rotation_flip(img)  # Apply random rotation and flip operations
      if segmentation:  # If segmentation is true
        img = apply_segmentation(img, threshold=65)  # Apply segmentation
      img_arr = img_to_array(img)  # Convert image to a array
      Images.append(img_arr)
      num_samples_succesfull += 1  # Increase the number of images that succesfully loaded

    except Exception as e:  # If any expection occurs
      num_samples_failed += 1  # Increase the number of images that failed to load
      print("[ERROR][#%d] " % (num_samples_failed), image_path, str(e))  # Print the path of the image and error itself
      continue  

  print("[INFO] %d samples loaded succesfully, %d samples failed to load.\n" % (num_samples_succesfull, num_samples_failed))
  Images = np.asarray(Images).astype('float32')  # Convert to Numpy array with float32 data type

  if configuration == 'pretrained':  # If the configuration is 'pretrained'
    Images = np.squeeze(Images, axis=-1)  # Remove last dimension of the image
    Images = np.repeat(Images[..., np.newaxis], 3, -1)  # Repeat current color channel for three times to make last dimension 3
    
  mean = np.mean(Images)  # Mean of the pixel values of images
  config.MEAN = mean
  std = np.std(Images)  # Standart deviation of the pixel values of images
  config.STD = std
  Images = (Images - mean) / std  # Normalization

  """
  if K.image_data_format() == "channels_first":
    Images = np.expand_dims(Images, axis=1)  # Extended dimension 1

  if K.image_data_format() == "channels_last":
    Images = np.expand_dims(Images, axis=-1)  # Extended dimension 3(usebackend tensorflow:aixs=3; theano:axixs=1) 
  """

  return Images


def random_rotation_flip(image):
  '''
  Performs random flip and rotation operations on given image.

  args:
  image -- Image file to perform operations.
  
  returns:
  image -- Image file applied flip and rotation operations.
  '''

  flip_methods = ['PIL.Image.FLIP_LEFT_RIGHT',  
                  'PIL.Image.FLIP_TOP_BOTTOM']

  rotation_methods = ['PIL.Image.ROTATE_90',
                      'PIL.Image.ROTATE_180',
                      'PIL.Image.ROTATE_270',
                      'PIL.Image.TRANSPOSE',
                      'PIL.Image.TRANSVERSE']

  if random.randint(0, 1):  # Apply flip operation with %50 possibility
    flip_method = random.choice(flip_methods)  # Choose a random method
    image = im.transpose(method=flip_method)  # Apply choosed method

  if random.randint(0, 1):  # Apply rotation operation with %50 possibility
    rotation_method = random.choice(rotation_methods)  # Choose a random method
    image = im.transpose(method=rotation_method)  # Apply choosed method

  return image


def apply_segmentation(image, threshold):
  '''
  Performs segmentation operation on given image.

  args:
  image -- Image file to perform operations.
  threshold --  Threshold value.

  returns:
  binary_image_array -- Image file applied segmentation operation.
  '''
  
  # Loop over all pixel values in image,
  # set pixel value to 0 if it is greater than `threshold`
  # otherwise, set the pixel value to 1
  binary_image = image.point(lambda p: p > threshold and 255) 

  return binary_image