# !/usr/bin/python
import config
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as layers
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPool2D, BatchNormalization, GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


def train(model, X_, y_, num_epochs=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE):
  
  # Define learning rate scheduler
  #lr_schedule = keras.optimizers.schedules.ExponentialDecay(
  #    INITIAL_LEARNING_RATE, decay_steps=100000, decay_rate=0.96, staircase=True
  #)

  # Compile the model
  model.compile(
      loss='categorical_crossentropy',  # Hata fonksiyonu olarak binary crossentropy seç
      optimizer=keras.optimizers.Adam(learning_rate=config.INITIAL_LEARNING_RATE),  # Adam optimizasyonu kullan
      metrics=['acc'],  # Doğruluk metriği kullan
  )

  # Save the models those has best validation acuracy
  checkpoint_cb = ModelCheckpoint(
      config.checkpoint_dir + 'model.{epoch:02d}-{val_acc:.2f}.h5', save_best_only=True
  )

  # Save the model history into a .CSV file
  csv_logger_cb = CSVLogger(config.csv_loggger_dir + 'training.log')

  # Train the model and validate at the end of each epoch
  history = model.fit(
      x=X_,  # Train data
      y=y_,  # Validation data
      batch_size=batch_size, 
      epochs=num_epochs,
      validation_split=0.10,  # Split %10 of train data as validation data
      shuffle=True,
      callbacks=[checkpoint_cb, csv_logger_cb],
      verbose=1,
  )

  return history


def decode_label(pred):
    #return 1 if score > 0.5 else 0
    return np.argmax(pred, axis=0)

