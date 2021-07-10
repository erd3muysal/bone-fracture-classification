# !/usr/bin/python
import load
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def show_image(image_path, report=True):
  '''
  Shows given image and reports details of it.

  args:
  image_path -- Path of image file to be showed.
  '''

  img = load.load_img(image_path[0], color_mode='grayscale')
  
  if report:  # If report is true
    # Print all image details
    print("Type of the image file: ", type(img))
    print("Format of the image file: ", img.format)
    print("Mode of the image file: ", img.mode)
    print("Size of the image file: ", img.size)

  plt.title("Label of the image: %d" %(image_path[1]))  # Show image label on title
  plt.imshow(img, cmap='gray')


def plot_image_slices(image_path, num_rows=4, num_columns=4, image_size=(8, 8)):
  '''
  Plot a montage of 16 (default) image slices.

  args:
  image_path -- Path of image file set to be showed.
  num_rows -- Number of rows.
  num_columns -- Number of columns.
  image_size -- Size of the image.
  '''

  w, h = image_size
  fig = plt.figure(figsize=(16, 16))

  for i in range(1, num_columns * num_rows + 1):
      fig.add_subplot(num_rows, num_columns, i)
      show_image(image_path[i], report=False)


def plot_confusion_matrix(y_test, y_pred):
  '''
  Karışıklık matrisini çizdirir.
  '''

  cm = confusion_matrix(y_test, y_pred)
  sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False, cmap="YlGnBu")
  plt.ylabel('Doğru etiket', fontsize=17)
  plt.xlabel('Tahmini etiket', fontsize=17)
  plt.show()


def plot_loss_acc(history, save=True):
  '''
  Her epokta ulaşılan hata ve başarım değerlerini çizdirir.
  '''

  figures_dir = os.path.join(os.getcwd(), 'figures/')
  if not os.path.isdir(figures_dir):
    os.makedirs(figures_dir)

  with open(os.path.join(os.getcwd(), 'log/training.log'), 'r') as f:
    log = f.read()

  fig, ax = plt.subplots(1, 2, figsize=(20, 3))
  ax = ax.ravel()
                               
  for i, metric in enumerate(['acc', 'loss']):
    ax[i].plot(history.history[metric])
    ax[i].plot(history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])
    ax[i].grid(True)

    if save:
      fig.savefig(figures_dir + metric + '_plot.jpg')

  plt.show()
  plt.clf()
  plt.close()